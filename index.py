"""
Duffel API - Proof-Slice Backend (FastAPI + Jinja2 Templates)
================================================================
Serves "travelers.html" (chatbot-style search UI) and exposes these endpoints:

    GET  /                       -> renders travelers.html
    POST /api/search             -> live flight search via Duffel (no auth needed - browsing is open)
    POST /api/chat               -> free-text message -> Claude decides whether to search
    POST /api/book                -> books the selected offer, returns the PNR (AUTH REQUIRED)
    POST /api/webhooks/duffel    -> receives order.created and
                                     order.airline_initiated_change_detected events from Duffel

    -- Auth (Section 5 - real Google OAuth + email/password) --
    POST /api/auth/signup         -> create an account with email + password + confirm_password
    POST /api/auth/login          -> log in with email + password
    POST /api/auth/logout         -> clear the logged-in session
    GET  /api/auth/me             -> tells the frontend whether someone is currently logged in
    GET  /api/auth/google/login   -> redirects the browser into Google's real OAuth consent screen
    GET  /api/auth/google/callback -> Google redirects back here after the user approves

Running in TEST MODE - no real money or real bookings involved.

Requires MongoDB running locally (or a MONGODB_URI pointing to one) - conversation
history is stored in the "traveling" database, "conversations" collection.
Confirmed bookings are stored in the "bookings" collection (Section 4e extension
below), and once a booking is confirmed, that session's conversation history is
archived into "conversation_archive" and then reset - so the next chat message
starts Claude on a clean slate instead of dragging along the whole completed
booking flow as context.

IMPORTANT - Duffel API access:
    This version talks to the Duffel REST API directly over HTTP (using the
    `requests` library) instead of the `duffel_api` PyPI package. That package
    is no longer maintained by Duffel and hardcodes an old `Duffel-Version`
    header, which Duffel has since discontinued - causing every request to
    fail with "Unsupported version". Talking to the REST API directly lets us
    control that header ourselves (see DUFFEL_VERSION below), so we can keep
    it up to date whenever Duffel ships a new version.

IMPORTANT - Auth model (Section 5):
    Browsing and searching flights (/api/search, /api/chat) stays OPEN - no
    login needed. The moment someone actually tries to book a flight
    (/api/book), a real logged-in account is required - either:
      a) Email + password (set at signup, confirmed with confirm_password), or
      b) "Sign in with Google" - a real Google OAuth 2.0 flow. When someone
         signs in this way there is no password to set/confirm at all; Google
         is the one that verified their identity.
    The frontend is expected to call GET /api/auth/me right after the
    traveler clicks/selects a flight offer, and if `authenticated` comes back
    false, show a sign-up/sign-in form (email+password+confirm, or a
    "Continue with Google" button) BEFORE moving on to the passenger-details
    step. /api/book itself also enforces this server-side, so booking can
    never happen without a real logged-in account even if the frontend check
    were skipped.

Setup:
    pip install requests python-dotenv fastapi uvicorn jinja2 python-multipart anthropic pymongo bcrypt itsdangerous --break-system-packages

Put your keys in a .env file:
    DUFFEL_ACCESS_TOKEN=duffel_test_XXXXXXXXXXXX
    ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXX
    MONGODB_URI=mongodb://localhost:27017
    CLAUDE_MODEL=claude-haiku-4-5-20251001
    DUFFEL_WEBHOOK_SECRET=whsec_XXXXXXXXXXXX

    # Real Google OAuth (Section 5) - create these in Google Cloud Console ->
    # APIs & Services -> Credentials -> OAuth client ID -> Web application.
    # Add GOOGLE_REDIRECT_URI below as an "Authorized redirect URI" there too.
    GOOGLE_CLIENT_ID=xxxxxxxxxx.apps.googleusercontent.com
    GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxx
    GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/api/auth/google/callback
    SESSION_SECRET_KEY=some-long-random-string-change-me

Run:
    uvicorn index:app --reload
    Then open: http://127.0.0.1:8000
"""

import os
import re
import uuid
import json
import hmac
import hashlib
import secrets
import urllib.parse
import datetime
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import bcrypt
import anthropic
from pymongo import MongoClient

load_dotenv()  # loads DUFFEL_ACCESS_TOKEN and ANTHROPIC_API_KEY from .env

app = FastAPI(title="AI Travel Assistant - Proof Slice")
templates = Jinja2Templates(directory="templates")  # travelers.html lives here

# ---------------------------------------------------------------------------
# Login session cookie (Section 5 - real auth)
#
# This is a signed, tamper-proof cookie (Starlette's SessionMiddleware, backed
# by itsdangerous) that holds request.session["user_email"] once someone logs
# in - either via email/password or via Google. It is a SEPARATE cookie from
# the anonymous "session_id" cookie used further down for chat/conversation
# tracking, so the two never collide.
# ---------------------------------------------------------------------------
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-only-insecure-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/auth/google/callback")
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

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


# A short lookup so we can point someone to a real, well-known city/IATA
# code when they type a country instead ("US", "UK", "Pakistan", ...)
# rather than just repeating Duffel's raw validation error. This is NOT
# meant to be exhaustive - anything not in here still gets a clear, generic
# tip instead of a confusing "invalid IATA code" message.
COUNTRY_CITY_HINTS = {
    "us": "New York (JFK), Los Angeles (LAX), or Miami (MIA)",
    "usa": "New York (JFK), Los Angeles (LAX), or Miami (MIA)",
    "united states": "New York (JFK), Los Angeles (LAX), or Miami (MIA)",
    "uk": "London (LHR) or Manchester (MAN)",
    "united kingdom": "London (LHR) or Manchester (MAN)",
    "pk": "Karachi (KHI), Lahore (LHE), or Islamabad (ISB)",
    "pakistan": "Karachi (KHI), Lahore (LHE), or Islamabad (ISB)",
    "uae": "Dubai (DXB) or Abu Dhabi (AUH)",
    "ae": "Dubai (DXB) or Abu Dhabi (AUH)",
    "saudi arabia": "Jeddah (JED) or Riyadh (RUH)",
    "sa": "Jeddah (JED) or Riyadh (RUH)",
    "turkey": "Istanbul (IST)",
    "tr": "Istanbul (IST)",
    "india": "Delhi (DEL) or Mumbai (BOM)",
    "in": "Delhi (DEL) or Mumbai (BOM)",
    "china": "Beijing (PEK) or Shanghai (PVG)",
    "cn": "Beijing (PEK) or Shanghai (PVG)",
}


def friendly_flight_input_error(raw_message: str, origin: str, destination: str) -> str:
    """
    Duffel rejects anything that isn't a real 3-letter IATA airport/city code
    with a fairly technical message, e.g. "Field 'destination' is invalid.
    Expected a valid IATA code." - which is exactly what happens when
    someone types a country ("US") instead of a specific city.

    This turns that into something a traveler can actually act on. It lives
    right here in one place and is called from inside search_flights()
    below, so BOTH /api/search (the manual form) and /api/chat (Claude's
    search_flights tool) get the exact same friendly wording for the exact
    same underlying problem - there's only one flight-search code path, so
    there's nothing to keep in sync between the two, and no extra Claude
    call is needed to produce this message either way.
    """
    bad_fields = re.findall(r"Field '(origin|destination)' is invalid", raw_message)
    if not bad_fields:
        return raw_message  # some other Duffel error - surface it as-is

    field_values = {"origin": origin, "destination": destination}
    field_labels = {"origin": "departure city", "destination": "destination city"}

    parts = []
    for field in bad_fields:
        value = field_values[field]
        label = field_labels[field]
        hint = COUNTRY_CITY_HINTS.get((value or "").strip().lower())
        if hint:
            parts.append(f"'{value}' looks like a country, not a specific {label} - try a city like {hint}.")
        else:
            parts.append(
                f"'{value}' isn't a valid {label}. Please use a specific city or its 3-letter "
                f"airport code (e.g. KHI for Karachi, IST for Istanbul)."
            )
    return " ".join(parts)


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
3. If the customer names a COUNTRY instead of a specific city (e.g. "I want to
   go to the US" or "US" as the destination), do NOT guess a city and do NOT
   call the search_flights tool - ask which city in that country they mean,
   optionally suggesting a couple of well-known ones (e.g. "Which city in the
   US - New York, Los Angeles, Miami?").
4. If the date or passenger count is missing, do NOT call the search_flights tool yet -
   ask the customer a short, friendly follow-up question in English first.
5. Once you have a specific origin city, destination city, and date, call the search_flights tool.
6. Always reply in English. Keep responses short and direct - no long explanations.
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
# Database: "traveling", Collections:
#   - "conversations"         {"session_id": ..., "history": [...], "updated_at": ...}
#   - "conversation_archive"  a copy of a session's history, saved right before
#                             it gets reset (see archive_and_reset_conversation)
#   - "bookings"              one document per confirmed booking (see below)
#   - "users"                 one document per account (Section 5 - real auth)
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["traveling"]
conversations = db["conversations"]
webhook_events = db["webhook_events"]  # log of every Duffel webhook event received
conversation_archive = db["conversation_archive"]  # snapshots of history right before a reset
bookings = db["bookings"]  # every confirmed booking, keyed by the logged-in account's email
users = db["users"]  # accounts - either email+password or Google-linked


def init_db():
    """Ensures session_id is indexed and unique, for fast lookups and no duplicates."""
    conversations.create_index("session_id", unique=True)
    conversation_archive.create_index("session_id")
    # Not unique - the same account can (and should be able to) book more than once.
    # This index is what lets us quickly answer "how many times has this exact
    # account booked with us before?" without scanning the whole collection.
    bookings.create_index("account_email")
    bookings.create_index("order_id", unique=True)
    users.create_index("email", unique=True)


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


def count_bookings_for_email(email: str) -> int:
    """How many confirmed bookings this exact logged-in account already has."""
    return bookings.count_documents({"account_email": email})


def archive_and_reset_conversation(session_id: str):
    """
    Called right after a booking is confirmed.

    Whatever conversation happened before this point (whether the traveler
    used the chat, the manual search-bar form, or a mix of both - both paths
    funnel through the same /api/book endpoint, so this behaves identically
    either way) gets snapshotted into conversation_archive for the record,
    and the *active* history for this session is reset to empty.

    That way the next message this session sends to /api/chat starts Claude
    on a fresh conversation instead of replaying the whole completed booking
    flow as context. Python (via the bookings collection / count_bookings_for_email)
    is what keeps track of how many times this traveler has booked - Claude
    itself doesn't need to carry that across chats.
    """
    history = load_history(session_id)
    if history:
        conversation_archive.insert_one({
            "session_id": session_id,
            "history": history,
            "archived_at": datetime.datetime.utcnow(),
        })
    save_history(session_id, [])


# ---------------------------------------------------------------------------
# Auth helpers (Section 5 - real Google OAuth + email/password)
# ---------------------------------------------------------------------------

def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    """bcrypt only looks at the first 72 bytes of a password - anything
    longer is silently ignored, same as most real-world auth systems."""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except ValueError:
        # Malformed/legacy hash - treat as a failed login rather than a 500.
        return False


def get_user_by_email(email: str):
    return users.find_one({"email": normalize_email(email)})


def create_local_account(email: str, password: str):
    """Email + password signup. Password is hashed with bcrypt - never stored in plain text."""
    doc = {
        "email": normalize_email(email),
        "password_hash": hash_password(password),
        "google_id": None,
        "name": None,
        "created_at": datetime.datetime.utcnow(),
    }
    users.insert_one(doc)
    return doc


def upsert_google_account(email: str, google_id: str, name: Optional[str]):
    """
    Signing in with Google verifies the email for us - no password is ever
    set or asked for on this path. If an email+password account already
    exists with this same email, we just link the Google id onto it so
    either method logs into the same account going forward.
    """
    email = normalize_email(email)
    existing = get_user_by_email(email)
    if existing:
        users.update_one({"_id": existing["_id"]}, {"$set": {"google_id": google_id, "name": name or existing.get("name")}})
        return get_user_by_email(email)
    doc = {
        "email": email,
        "password_hash": None,
        "google_id": google_id,
        "name": name,
        "created_at": datetime.datetime.utcnow(),
    }
    users.insert_one(doc)
    return doc


def current_user_email(request: Request) -> Optional[str]:
    """The logged-in account's email, or None if nobody is logged in on this browser."""
    return request.session.get("user_email")


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
    # Full traveler list for multi-passenger bookings. The frontend always
    # sends this alongside `passenger` (which is just passengers[0], kept
    # for backward compatibility). If it's ever missing/empty for some
    # caller, we fall back to booking just the one `passenger` above.
    passengers: List[PassengerDetails] = []


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []  # prior conversation, in Anthropic message format


class SignupRequest(BaseModel):
    email: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# ---------------------------------------------------------------------------
# STEP 1 - Flight Search (the first step of aggregation)
# ---------------------------------------------------------------------------

def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    """
    Sends an 'offer_request' to Duffel via direct HTTP call - i.e. tells it
    "give me options for this route, this date, this many passengers"

    Returns: top 3 offers (as plain dicts), sorted by price (cheapest first)
    """
    body = {
        "data": {
            "slices": [{
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
            }],
            "passengers": [{"type": "adult"} for _ in range(passengers)],
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
        raise Exception(friendly_flight_input_error(duffel_error_message(resp), origin, destination))

    offer_request = resp.json()["data"]
    offers = offer_request.get("offers", [])
    offers_sorted = sorted(offers, key=lambda o: float(o["total_amount"]))
    return offers_sorted[:3]  # only show top 3, as specified in the brief


# ---------------------------------------------------------------------------
# STEP 2 - Create the Booking (Order)
# ---------------------------------------------------------------------------

def book_flight(offer_id: str, passengers_details: list):
    """
    Uses payment type "balance" - meaning it's deducted automatically from the
    agency's Duffel wallet (which was topped up earlier in test mode).

    `passengers_details` is a list with one dict per traveler - it must have
    exactly as many entries as the offer itself has passenger slots (e.g. a
    2-passenger search produces an offer with 2 passenger slots, so this
    needs exactly 2 entries). Duffel rejects the order otherwise with
    "Field 'passengers' should have N item(s)".
    """
    # Fetch the offer first, to get its passenger ids, current price, and currency
    resp = requests.get(
        f"{DUFFEL_API_BASE}/air/offers/{offer_id}",
        headers=DUFFEL_HEADERS,
        params={"return_available_services": "true"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise Exception(duffel_error_message(resp))
    offer = resp.json()["data"]

    offer_passengers = offer["passengers"]
    if len(passengers_details) != len(offer_passengers):
        raise Exception(
            f"This flight was searched for {len(offer_passengers)} passenger(s), "
            f"but {len(passengers_details)} traveler(s) were submitted. "
            f"Please provide details for all {len(offer_passengers)} passenger(s)."
        )

    order_passengers = [
        {"id": offer_passengers[i]["id"], **passengers_details[i]}
        for i in range(len(offer_passengers))
    ]

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
    """Called when the traveler searches using the manual form. No login required - browsing is open."""
    try:
        offers = search_flights(req.origin, req.destination, req.departure_date, req.passengers)
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
      - If info is missing (date/passengers) -> asks first, does NOT call the tool

    Python is only the "hands" here - the decision always belongs to Claude.

    Conversation history is loaded from and saved to MongoDB, keyed by a
    session_id cookie - so it survives page reloads/server restarts, and two
    different users' conversations can never mix up with each other.

    Note: right after a booking is confirmed via /api/book, this session's
    stored history gets archived and reset to empty (see
    archive_and_reset_conversation) - so if the traveler chats again after
    booking, they start a brand new conversation with Claude instead of the
    whole finished booking flow being replayed as context.
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
            passengers=args.get("passengers", 1),
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


# ---------------------------------------------------------------------------
# Auth routes (Section 5 - real Google OAuth + email/password)
#
# Searching stays open. The moment a traveler clicks a flight offer, the
# frontend should call GET /api/auth/me - if authenticated is false, show
# the sign-up/sign-in form (email + password + confirm_password, or a
# "Continue with Google" button) before letting them go on to passenger
# details. /api/book (further below) also checks this itself.
# ---------------------------------------------------------------------------

@app.get("/api/auth/me")
def api_auth_me(request: Request):
    """Lets the frontend check, right after a flight is clicked, whether a login is needed."""
    email = current_user_email(request)
    return {"authenticated": bool(email), "email": email}


@app.post("/api/auth/signup")
def api_auth_signup(req: SignupRequest, request: Request):
    """Email + password signup. Requires password and confirm_password to match."""
    email = normalize_email(req.email)

    if req.password != req.confirm_password:
        return {"success": False, "error": "Password and confirm password do not match."}
    if len(req.password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}
    if get_user_by_email(email):
        return {"success": False, "error": "An account with this email already exists. Please log in instead."}

    create_local_account(email, req.password)
    request.session["user_email"] = email
    return {"success": True, "email": email}


@app.post("/api/auth/login")
def api_auth_login(req: LoginRequest, request: Request):
    """Email + password login for accounts that were created with a password."""
    user = get_user_by_email(req.email)
    if not user or not user.get("password_hash") or not verify_password(req.password, user["password_hash"]):
        return {"success": False, "error": "Incorrect email or password."}

    request.session["user_email"] = user["email"]
    return {"success": True, "email": user["email"]}


@app.post("/api/auth/logout")
def api_auth_logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/auth/google/login")
def api_auth_google_login(request: Request):
    """
    Kicks off the REAL Google OAuth 2.0 flow - sends the browser to Google's
    own consent screen. Nothing about the user's Google password ever passes
    through this server; Google authenticates them directly.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Google OAuth is not configured - set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env",
        })

    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}")


@app.get("/api/auth/google/callback")
def api_auth_google_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """
    Google redirects the browser back here after the user approves (or
    cancels) the sign-in. We swap the one-time `code` for real tokens
    server-to-server, then ask Google who this is, and log them in - no
    password is ever set or needed for this path.
    """
    if error or not code:
        return RedirectResponse("/?auth_error=google_denied")

    if not state or state != request.session.get("oauth_state"):
        return RedirectResponse("/?auth_error=state_mismatch")
    request.session.pop("oauth_state", None)

    token_resp = requests.post(
        GOOGLE_TOKEN_ENDPOINT,
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if token_resp.status_code >= 400:
        return RedirectResponse("/?auth_error=token_exchange_failed")

    access_token = token_resp.json().get("access_token")
    userinfo_resp = requests.get(
        GOOGLE_USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if userinfo_resp.status_code >= 400:
        return RedirectResponse("/?auth_error=userinfo_failed")

    info = userinfo_resp.json()
    email = info.get("email")
    google_id = info.get("sub")
    name = info.get("name")
    if not email or not google_id:
        return RedirectResponse("/?auth_error=missing_profile")

    upsert_google_account(email, google_id, name)
    request.session["user_email"] = normalize_email(email)
    return RedirectResponse("/?auth=success")


@app.post("/api/book")
def api_book(req: BookRequest, request: Request, response: Response):
    """
    Called when the traveler submits the passenger form to confirm booking -
    whether they got to this point via the chat conversation or straight from
    the manual search-bar form. Both paths land here the same way, so
    everything below applies identically no matter which one was used.

    AUTH REQUIRED: booking can only go through for a logged-in account
    (email+password, or Google). This is enforced here server-side even
    though the frontend should already be gating the flow right after the
    flight was clicked.
    """
    account_email = current_user_email(request)
    if not account_email:
        return JSONResponse(status_code=401, content={
            "success": False,
            "error": "Please sign in (or continue with Google) to complete your booking.",
        })

    session_id = get_session_id(request, response)
    # Prefer the full `passengers` list (multi-passenger bookings); fall back
    # to the single `passenger` field only if the caller didn't send a list.
    passenger_list = [p.model_dump() for p in req.passengers] if req.passengers else [req.passenger.model_dump()]
    try:
        order = book_flight(req.offer_id, passenger_list)

        # --- Persist the booking itself (Section 4e extension) ---
        # How many bookings this exact logged-in account already had BEFORE
        # this one - this is how Python (not Claude) knows this traveler is a
        # repeat customer if/when they come back later, even in a brand new
        # chat session, since it's tied to their real account, not just a
        # cookie.
        prior_bookings_for_account = count_bookings_for_email(account_email)
        bookings.insert_one({
            "session_id": session_id,
            "account_email": account_email,
            "email": req.passenger.email,
            "order_id": order["id"],
            "pnr": order["booking_reference"],
            "offer_id": req.offer_id,
            "passenger": req.passenger.model_dump(),
            "passengers": passenger_list,
            "total_paid": f"{order['total_amount']} {order['total_currency']}",
            "booking_number_for_account": prior_bookings_for_account + 1,
            "booked_at": datetime.datetime.utcnow(),
        })

        # --- Archive this session's chat history, then start a clean slate ---
        archive_and_reset_conversation(session_id)

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
