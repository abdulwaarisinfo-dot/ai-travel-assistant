"""
Duffel API - Proof-Slice Backend (FastAPI + Jinja2 Templates)
================================================================
Serves "travelers.html" (chatbot-style search UI) and exposes these endpoints:

    GET  /                      -> renders travelers.html
    POST /api/search            -> live flight search via Duffel (adults + children)
    POST /api/chat              -> free-text message -> Claude decides whether to search
    POST /api/book               -> books the selected offer for ALL passengers
                                    (adults + children), returns the PNR
    POST /api/webhooks/duffel   -> receives order.created and
                                    order.airline_initiated_change_detected events from Duffel

Running in TEST MODE - no real money or real bookings involved.

Requires MongoDB running locally (or a MONGODB_URI pointing to one) - conversation
history is stored in the "traveling" database, "conversations" collection.

IMPORTANT - Duffel API access:
    This version talks to the Duffel REST API directly over HTTP (using the
    `requests` library) instead of the `duffel_api` PyPI package. That package
    is no longer maintained by Duffel and hardcodes an old `Duffel-Version`
    header, which Duffel has since discontinued - causing every request to
    fail with "Unsupported version". Talking to the REST API directly lets us
    control that header ourselves (see DUFFEL_VERSION below), so we can keep
    it up to date whenever Duffel ships a new version.

IMPORTANT - Adults / children:
    Real travelers rarely search "3 passengers" - they search "2 adults, 1 child".
    So /api/search and Claude's search_flights tool now take separate `adults`
    and `children` counts instead of one flat `passengers` number. Duffel builds
    a distinct offer "passenger slot" (with its own id) for every adult and every
    child, and each slot's type (adult/child) is fixed at search time.

    Because of that, booking a multi-passenger offer can't just take ONE
    passenger's details anymore - it needs one passenger's details PER slot,
    with each one correctly matched back to an adult or child slot on the offer
    (see book_flight() below for how that matching is done).

Setup:
    pip install requests python-dotenv fastapi uvicorn jinja2 python-multipart anthropic pymongo --break-system-packages

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
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Literal
import anthropic
from pymongo import MongoClient

load_dotenv()  # loads DUFFEL_ACCESS_TOKEN and ANTHROPIC_API_KEY from .env

app = FastAPI(title="AI Travel Assistant - Proof Slice")
templates = Jinja2Templates(directory="templates")  # travelers.html lives here

# ---------------------------------------------------------------------------
# Duffel REST API Setup (direct HTTP - no duffel_api package)
# ---------------------------------------------------------------------------
DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"  # bump this if Duffel ships a new API version in future
DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")

DUFFEL_HEADERS = {
    "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
    "Duffel-Version": DUFFEL_VERSION,
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
}


def duffel_error_message(resp: requests.Response) -> str:
    """Pulls a readable error message out of a Duffel error response body."""
    try:
        errors = resp.json().get("errors", [])
        if errors:
            return errors[0].get("message", resp.text)
    except Exception:
        pass
    return f"Duffel API error (status {resp.status_code}): {resp.text}"


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
1. Extract the route (origin/destination city), travel date, and the number of
   adult and child passengers. "Child" means a passenger roughly 2-11 years old
   travelling with their own seat. If the customer just says "2 people" or
   "3 passengers" with no mention of children, assume they are all adults.
2. Convert city names to their 3-letter IATA airport codes yourself
   (e.g. Lahore=LHR, Islamabad=ISB, Karachi=KHI, Istanbul=IST, Dubai=DXB).
3. If the date is missing, do NOT call the search_flights tool yet -
   ask the customer a short, friendly follow-up question in English first.
4. Once you have origin, destination, and date, call the search_flights tool
   with the adults and children counts you've gathered (default adults=1,
   children=0 if the customer never mentions passenger counts at all).
5. Always reply in English. Keep responses short and direct - no long explanations.
6. Do NOT use Markdown formatting (no **bold**, no numbered/bulleted lists with - or *,
   no headers). The chat UI displays your reply as plain text, so Markdown symbols would
   show up literally instead of being styled. Write plain sentences instead, e.g. use
   "1)" or simple line breaks for lists if needed.
"""

SEARCH_TOOL = {
    "name": "search_flights",
    "description": "Searches for flights between two airports on a specific date, for a given number of adult and child passengers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "3-letter IATA code, departure city"},
            "destination": {"type": "string", "description": "3-letter IATA code, arrival city"},
            "departure_date": {"type": "string", "description": "Format: YYYY-MM-DD"},
            "adults": {"type": "integer", "description": "Number of adult passengers (age 12+)", "default": 1},
            "children": {"type": "integer", "description": "Number of child passengers (roughly age 2-11)", "default": 0},
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
users = db["users"]  # DEMO ONLY - dummy account records for the signup flow (not real auth)


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
    adults: int = 1
    children: int = 0


class PassengerDetails(BaseModel):
    given_name: str
    family_name: str
    email: str
    phone_number: str
    born_on: str
    gender: str = "m"
    title: str = "mr"
    # Which slot this passenger fills on the offer - "adult" or "child".
    # NOT sent to Duffel directly (see book_flight below); it's only used
    # here to match each set of details back to the right passenger id.
    type: Literal["adult", "child"] = "adult"


class SignupRequest(BaseModel):
    email: str
    phone_number: str
    password: str


class BookRequest(BaseModel):
    offer_id: str
    passengers: List[PassengerDetails]  # one entry per adult/child on the offer


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []  # prior conversation, in Anthropic message format


# ---------------------------------------------------------------------------
# STEP 1 - Flight Search (the first step of aggregation)
# ---------------------------------------------------------------------------

def search_flights(origin: str, destination: str, departure_date: str, adults: int = 1, children: int = 0):
    """
    Sends an 'offer_request' to Duffel via direct HTTP call - i.e. tells it
    "give me options for this route, this date, for this many adults and children"

    Duffel creates one passenger "slot" per adult and per child, each with its
    own type - those slots (and their ids) are what we later match real
    passenger details to when booking.

    Returns: top 3 offers (as plain dicts), sorted by price (cheapest first)
    """
    passengers_payload = (
        [{"type": "adult"} for _ in range(adults)]
        + [{"type": "child"} for _ in range(children)]
    )

    body = {
        "data": {
            "slices": [{
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
            }],
            "passengers": passengers_payload,
            "cabin_class": "economy",
        }
    }

    resp = requests.post(
        f"{DUFFEL_API_BASE}/air/offer_requests",
        headers=DUFFEL_HEADERS,
        params={"return_offers": "true"},
        json=body,
        timeout=30,
    )
    if resp.status_code >= 400:
        raise Exception(duffel_error_message(resp))

    offer_request = resp.json()["data"]
    offers = offer_request.get("offers", [])
    offers_sorted = sorted(offers, key=lambda o: float(o["total_amount"]))
    return offers_sorted[:3]  # only show top 3, as specified in the brief


# ---------------------------------------------------------------------------
# STEP 2 - Create the Booking (Order)
# ---------------------------------------------------------------------------

def book_flight(offer_id: str, passengers_details: List[dict]):
    """
    Books every passenger on the offer - not just one. Each offer has a fixed
    list of passenger "slots" (one per adult/child requested at search time),
    each with its own id and type ("adult" or "child"). We match the details
    submitted from the booking form back to those slots BY TYPE, not by
    position, so it doesn't matter what order the form filled them in:

        offer's adult slots  <- zipped with -> submitted adult passengers
        offer's child slots  <- zipped with -> submitted child passengers

    Uses payment type "balance" - meaning it's deducted automatically from the
    agency's Duffel wallet (which was topped up earlier in test mode).
    """
    # Fetch the offer first, to get its passenger ids/types, current price, and currency
    resp = requests.get(
        f"{DUFFEL_API_BASE}/air/offers/{offer_id}",
        headers=DUFFEL_HEADERS,
        params={"return_available_services": "true"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise Exception(duffel_error_message(resp))
    offer = resp.json()["data"]

    offer_adult_ids = [p["id"] for p in offer["passengers"] if p.get("type") == "adult"]
    offer_child_ids = [p["id"] for p in offer["passengers"] if p.get("type") == "child"]

    submitted_adults = [p for p in passengers_details if p.get("type", "adult") == "adult"]
    submitted_children = [p for p in passengers_details if p.get("type") == "child"]

    if len(submitted_adults) != len(offer_adult_ids) or len(submitted_children) != len(offer_child_ids):
        raise Exception(
            f"Passenger details don't match this offer: offer needs "
            f"{len(offer_adult_ids)} adult(s) and {len(offer_child_ids)} child(ren), "
            f"but {len(submitted_adults)} adult and {len(submitted_children)} child "
            f"passenger detail(s) were submitted."
        )

    order_passengers = []
    for offer_id_slot, details in zip(offer_adult_ids, submitted_adults):
        details = {k: v for k, v in details.items() if k != "type"}
        order_passengers.append({"id": offer_id_slot, **details})
    for offer_id_slot, details in zip(offer_child_ids, submitted_children):
        details = {k: v for k, v in details.items() if k != "type"}
        order_passengers.append({"id": offer_id_slot, **details})

    order_body = {
        "data": {
            "selected_offers": [offer_id],
            "payments": [{
                "type": "balance",
                "currency": offer["total_currency"],
                "amount": offer["total_amount"],
            }],
            "passengers": order_passengers,
        }
    }

    resp = requests.post(
        f"{DUFFEL_API_BASE}/air/orders",
        headers=DUFFEL_HEADERS,
        json=order_body,
        timeout=30,
    )
    if resp.status_code >= 400:
        raise Exception(duffel_error_message(resp))

    return resp.json()["data"]


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
        offers = search_flights(req.origin, req.destination, req.departure_date, req.adults, req.children)
        results = [
            {
                "offer_id": o["id"],
                "airline": o["owner"]["name"],
                "price": o["total_amount"],
                "currency": o["total_currency"],
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
      - If info is missing (date) -> asks first, does NOT call the tool

    Python is only the "hands" here - the decision always belongs to Claude.

    Conversation history is loaded from and saved to MongoDB, keyed by a
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
        updated_history = messages + [{"role": "assistant", "content": [b.model_dump() for b in response_msg.content]}]
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
            adults=args.get("adults", 1),
            children=args.get("children", 0),
        )
        results = [
            {"offer_id": o["id"], "airline": o["owner"]["name"], "price": o["total_amount"], "currency": o["total_currency"]}
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
        {"role": "assistant", "content": [b.model_dump() for b in response_msg.content]},
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
    updated_history = follow_up_messages + [{"role": "assistant", "content": [b.model_dump() for b in final_response.content]}]
    save_history(session_id, updated_history)

    return {
        "success": True,
        "reply": reply_text,
        "offers": results if results else None,
        "history": updated_history,
    }


@app.post("/api/signup")
def api_signup(req: SignupRequest):
    """
    DEMO ONLY - creates a dummy account record so the booking flow feels like
    a real travel site (sign up -> fill passenger details -> pay -> confirm).
    This is NOT a real authentication system: there's no login, no session
    tied to it, and it isn't used to authorize the booking. It exists purely
    so the agency owner sees the full realistic flow during the demo.

    The password is hashed before storage (never stored in plain text) as
    basic good practice, even though this isn't a production auth system.
    """
    try:
        password_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest()
        users.update_one(
            {"email": req.email},
            {"$set": {
                "email": req.email,
                "phone_number": req.phone_number,
                "password_hash": password_hash,
                "created_at": datetime.datetime.utcnow(),
            }},
            upsert=True,
        )
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/book")
def api_book(req: BookRequest):
    """
    Called when the traveler submits the passenger form(s) to confirm booking.
    Now takes ONE set of passenger details PER adult/child on the offer
    (req.passengers is a list), instead of assuming a single traveler.
    """
    try:
        order = book_flight(req.offer_id, [p.model_dump() for p in req.passengers])
        return {
            "success": True,
            "pnr": order["booking_reference"],
            "order_id": order["id"],
            "total_paid": f"{order['total_amount']} {order['total_currency']}",
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

    # Duffel sends the header as: t=<timestamp>,v2=<signature>
    # (Note: Duffel API v2 uses "v2=" here, not the older "v1=" scheme.)
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(","))
        timestamp = parts["t"]
        received_sig = parts["v2"]
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
