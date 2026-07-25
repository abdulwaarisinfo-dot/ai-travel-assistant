"""
Duffel API - Proof-Slice Backend (FastAPI + Jinja2 Templates)
================================================================
Serves "travelers.html" (chatbot-style search UI) and exposes these endpoints:

    GET  /                      -> renders travelers.html
    POST /api/search            -> live flight search via Duffel
    POST /api/chat              -> free-text message -> Claude decides whether to search
    POST /api/book               -> books the selected offer, returns the PNR
    POST /api/webhooks/duffel   -> receives order.created and
                                    order.airline_initiated_change_detected events from Duffel

Running in TEST MODE - no real money or real bookings involved.

Requires MongoDB running locally (or a MONGODB_URI pointing to one) - conversation
history is stored in the "traveling" database, "conversations" collection.

Setup:
    pip install duffel_api python-dotenv fastapi uvicorn jinja2 python-multipart anthropic pymongo --break-system-packages

Put your keys in a .env file:
    DUFFEL_ACCESS_TOKEN=duffel_test_XXXXXXXXXXXX
    ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXX
    MONGODB_URI=mongodb://localhost:27017
    CLAUDE_MODEL=claude-haiku-4-5-20251001
    DUFFEL_WEBHOOK_SECRET=whsec_XXXXXXXXXXXX

Run:
    uvicorn index:app --reload
    Then open: http://127.0.0.1:8000
"""

import os
import uuid
import json
import hmac
import hashlib
import datetime
from dotenv import load_dotenv
from duffel_api import Duffel
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import anthropic
from pymongo import MongoClient

load_dotenv()  # loads DUFFEL_ACCESS_TOKEN and ANTHROPIC_API_KEY from .env

app = FastAPI(title="AI Travel Assistant - Proof Slice")
templates = Jinja2Templates(directory="templates")  # travelers.html lives here

# --- Duffel Client Setup ---
# Never hardcode the token in code - always read it from an environment variable
client = Duffel(access_token=os.getenv("DUFFEL_ACCESS_TOKEN"))

# --- Duffel Webhook Setup ---
# Signing secret shown by Duffel when you register the webhook endpoint in
# their dashboard. Used to verify incoming webhook requests are genuinely
# from Duffel (see duffel_webhook() route below).
DUFFEL_WEBHOOK_SECRET = os.getenv("DUFFEL_WEBHOOK_SECRET")

# --- Claude Client Setup (Section 4d - AI Orchestration) ---
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")  # cheap/fast - enough for this task

SYSTEM_PROMPT = """You are an AI assistant for a travel agency. Customers write free text
in English (e.g. "I want to go from Lahore to Islamabad").

Your job:
1. Extract the route (origin/destination city), travel date, and passenger count.
2. Convert city names to their 3-letter IATA airport codes yourself
   (e.g. Lahore=LHR, Islamabad=ISB, Karachi=KHI, Istanbul=IST, Dubai=DXB).
3. If the date or passenger count is missing, do NOT call the search_flights tool yet -
   ask the customer a short, friendly follow-up question in English first.
4. Once you have origin, destination, and date, call the search_flights tool.
5. Always reply in English. Keep responses short and direct - no long explanations.
"""

SEARCH_TOOL = {
    "name": "search_flights",
    "description": "Searches for flights between two airports on a specific date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "3-letter IATA code, departure city"},
            "destination": {"type": "string", "description": "3-letter IATA code, arrival city"},
            "departure_date": {"type": "string", "description": "Format: YYYY-MM-DD"},
            "passengers": {"type": "integer", "description": "Number of adult passengers", "default": 1},
        },
        "required": ["origin", "destination", "departure_date"],
    },
}


# ---------------------------------------------------------------------------
# Global error handling - THIS FIXES the "Unexpected token '<'" bug.
# Without this, an unhandled exception can make FastAPI/Starlette return an
# HTML error page instead of JSON, and the frontend's res.json() call then
# crashes trying to parse HTML as JSON. This guarantees every response,
# even on a crash, is valid JSON.
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def all_exceptions_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Persistent Conversation Storage (Section 4e - Data Layer)
#
# Real MongoDB (NoSQL) storage so that:
#   1. Conversation history survives a server restart (not just kept in RAM)
#   2. Each browser/user gets its own session_id cookie -> conversations
#      from different people can NEVER get mixed up with each other
#
# Database: "traveling", Collection: "conversations"
# Each document looks like: {"session_id": "...", "history": [...], "updated_at": ...}
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["traveling"]
conversations = db["conversations"]
webhook_events = db["webhook_events"]  # log of every Duffel webhook event received


def init_db():
    """Ensures session_id is indexed and unique, for fast lookups and no duplicates."""
    conversations.create_index("session_id", unique=True)


def load_history(session_id: str) -> list:
    doc = conversations.find_one({"session_id": session_id})
    return doc["history"] if doc else []


def save_history(session_id: str, history: list):
    conversations.update_one(
        {"session_id": session_id},
        {"$set": {"history": history, "updated_at": datetime.datetime.utcnow()}},
        upsert=True,
    )


def get_session_id(request: Request, response: Response) -> str:
    """
    Reads the session_id cookie if present, otherwise creates a new one.
    httponly=True means JavaScript can't read/tamper with it - only the
    browser sends it back automatically on every request to this backend.
    """
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    return session_id


init_db()


# ---------------------------------------------------------------------------
# Request/Response Models (the shape of what the traveler types/submits)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    origin: str
    destination: str
    departure_date: str
    passengers: int = 1


class PassengerDetails(BaseModel):
    given_name: str
    family_name: str
    email: str
    phone_number: str
    born_on: str
    gender: str = "m"
    title: str = "mr"


class BookRequest(BaseModel):
    offer_id: str
    passenger: PassengerDetails


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []  # prior conversation, in Anthropic message format


# ---------------------------------------------------------------------------
# STEP 1 - Flight Search (the first step of aggregation)
# ---------------------------------------------------------------------------

def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    """
    Sends an 'offer_request' to Duffel - i.e. tells it
    "give me options for this route, this date, this many passengers"

    Note: the duffel_api library uses a builder/chaining pattern here
    (create() -> .slices(...) -> .passengers(...) -> .execute()), not plain
    keyword arguments - that mismatch was the cause of the earlier
    "unexpected keyword argument 'slices'" error.

    Returns: top 3 offers, sorted by price (cheapest first)
    """
    offer_request = (
        client.offer_requests.create()
        .slices([{
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
        }])
        .passengers([{"type": "adult"} for _ in range(passengers)])
        .cabin_class("economy")
        .return_offers()  # without this, offer_request.offers comes back empty
        .execute()
    )

    offers = sorted(offer_request.offers, key=lambda o: float(o.total_amount))
    return offers[:3]  # only show top 3, as specified in the brief


# ---------------------------------------------------------------------------
# STEP 2 - Create the Booking (Order)
# ---------------------------------------------------------------------------

def book_flight(offer_id: str, passenger_details: dict):
    """
    Uses payment type "balance" - meaning it's deducted automatically from the
    agency's Duffel wallet (which was topped up earlier in test mode).

    Same builder pattern as search_flights: create() -> .selected_offers(...)
    -> .payments(...) -> .passengers(...) -> .execute().
    """
    offer = client.offers.get(offer_id, return_available_services=True)

    order = (
        client.orders.create()
        .selected_offers([offer_id])
        .payments([{
            "type": "balance",
            "currency": offer.total_currency,
            "amount": offer.total_amount,
        }])
        .passengers([{
            "id": offer.passengers[0].id,
            **passenger_details,
        }])
        .execute()
    )
    return order


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the chatbot UI (templates/travelers.html)"""
    return templates.TemplateResponse("travelers.html", {"request": request})


@app.post("/api/search")
def api_search(req: SearchRequest):
    """Called when the traveler searches using the manual form"""
    try:
        offers = search_flights(req.origin, req.destination, req.departure_date, req.passengers)
        results = [
            {
                "offer_id": o.id,
                "airline": o.owner.name,
                "price": o.total_amount,
                "currency": o.total_currency,
            }
            for o in offers
        ]
        return {"success": True, "offers": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/chat")
def api_chat(req: ChatRequest, request: Request, response: Response):
    """
    AI Orchestration (Section 4d) + Persistent Session Storage (Section 4e)

    Takes a free-text message (e.g. "I want to go from Lahore to Islamabad")
    and sends it to Claude. Claude ITSELF decides:
      - If the info is complete -> calls the search_flights tool
      - If info is missing (date/passengers) -> asks first, does NOT call the tool

    Python is only the "hands" here - the decision always belongs to Claude.

    Conversation history is now loaded from and saved to SQLite, keyed by a
    session_id cookie - so it survives page reloads/server restarts, and two
    different users' conversations can never mix up with each other.
    """
    session_id = get_session_id(request, response)
    stored_history = load_history(session_id)

    messages = stored_history + [{"role": "user", "content": req.message}]

    try:
        response_msg = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
    except Exception as e:
        return {"success": False, "error": f"Claude API error: {e}"}

    # Case A - Claude did NOT call the tool yet (info missing, or just replying)
    if response_msg.stop_reason != "tool_use":
        reply_text = "".join(b.text for b in response_msg.content if b.type == "text")
        updated_history = messages + [{"role": "assistant", "content": response_msg.content}]
        save_history(session_id, updated_history)
        return {"success": True, "reply": reply_text, "offers": None, "history": updated_history}

    # Case B - Claude decided to run "search_flights"
    tool_use_block = next(b for b in response_msg.content if b.type == "tool_use")
    args = tool_use_block.input

    try:
        offers = search_flights(
            origin=args["origin"],
            destination=args["destination"],
            departure_date=args["departure_date"],
            passengers=args.get("passengers", 1),
        )
        results = [
            {"offer_id": o.id, "airline": o.owner.name, "price": o.total_amount, "currency": o.total_currency}
            for o in offers
        ]
        tool_result_content = {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": str(results) if results else "No flights found for this route/date.",
        }
    except Exception as e:
        results = []
        tool_result_content = {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": f"Search API error: {e}",
            "is_error": True,
        }

    # Send the tool result back to Claude so it can write a reply for the customer
    follow_up_messages = messages + [
        {"role": "assistant", "content": response_msg.content},
        {"role": "user", "content": [tool_result_content]},
    ]
    final_response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        tools=[SEARCH_TOOL],
        messages=follow_up_messages,
    )
    reply_text = "".join(b.text for b in final_response.content if b.type == "text")
    updated_history = follow_up_messages + [{"role": "assistant", "content": final_response.content}]
    save_history(session_id, updated_history)

    return {
        "success": True,
        "reply": reply_text,
        "offers": results if results else None,
        "history": updated_history,
    }


@app.post("/api/book")
def api_book(req: BookRequest):
    """Called when the traveler submits the passenger form to confirm booking"""
    try:
        order = book_flight(req.offer_id, req.passenger.model_dump())
        return {
            "success": True,
            "pnr": order.booking_reference,
            "order_id": order.id,
            "total_paid": f"{order.total_amount} {order.total_currency}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Webhook - Receives events from Duffel (order.created,
# order.airline_initiated_change_detected)
# ---------------------------------------------------------------------------

@app.post("/api/webhooks/duffel")
async def duffel_webhook(request: Request):
    """
    Duffel calls this URL whenever a subscribed event happens
    (order.created, order.airline_initiated_change_detected).

    Security: Duffel signs every webhook request with your webhook secret.
    We must read the RAW request body (not FastAPI's parsed JSON) because
    the signature is computed over the exact bytes Duffel sent - if we let
    FastAPI parse and reserialize the body first, the bytes could differ
    slightly and the signature check would fail.
    """
    raw_body = await request.body()
    signature_header = request.headers.get("X-Duffel-Signature", "")

    if not DUFFEL_WEBHOOK_SECRET:
        return JSONResponse(status_code=500, content={"success": False, "error": "Webhook secret not configured"})

    # Duffel sends the header as: t=<timestamp>,v1=<signature>
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(","))
        timestamp = parts["t"]
        received_sig = parts["v1"]
    except Exception:
        return JSONResponse(status_code=401, content={"success": False, "error": "Malformed signature header"})

    signed_payload = f"{timestamp}.{raw_body.decode('utf-8')}"
    expected_sig = hmac.new(
        DUFFEL_WEBHOOK_SECRET.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to avoid timing attacks
    if not hmac.compare_digest(expected_sig, received_sig):
        # TEMPORARY DEBUG LOGGING - remove once the 401 issue is resolved
        print(f"[webhook-debug] secret loaded: {bool(DUFFEL_WEBHOOK_SECRET)}, secret length: {len(DUFFEL_WEBHOOK_SECRET or '')}")
        print(f"[webhook-debug] secret fingerprint: {(DUFFEL_WEBHOOK_SECRET or '')[:4]}...{(DUFFEL_WEBHOOK_SECRET or '')[-4:]}")
        print(f"[webhook-debug] received signature header: {signature_header}")
        print(f"[webhook-debug] expected: {expected_sig}")
        print(f"[webhook-debug] received: {received_sig}")
        return JSONResponse(status_code=401, content={"success": False, "error": "Invalid signature"})

    # --- Signature verified - safe to process the event now ---
    payload = json.loads(raw_body)
    event_type = payload.get("type")
    event_data = payload.get("data", {})

    # Log every event received - useful for debugging and auditing
    webhook_events.insert_one({
        "type": event_type,
        "data": event_data,
        "received_at": datetime.datetime.utcnow(),
    })

    if event_type == "order.created":
        order_id = event_data.get("object_id") or event_data.get("id")
        print(f"[webhook] Booking confirmed: order {order_id}")
        # TODO: mark this order as confirmed in your own bookings collection

    elif event_type == "order.airline_initiated_change_detected":
        order_id = event_data.get("object_id") or event_data.get("id")
        print(f"[webhook] Airline changed the schedule for order {order_id}")
        # TODO: notify the customer (email/SMS) that their flight changed

    # Always return 200 quickly - Duffel retries if it doesn't get a fast 2xx response
    return {"success": True}
