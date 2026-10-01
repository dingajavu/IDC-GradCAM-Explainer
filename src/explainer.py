import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_TEMPLATE = """You are assisting a clinician reviewing an automated \
breast histopathology classifier's output. You have been given the model's \
prediction and Grad-CAM activation details below.

IMPORTANT CONSTRAINTS on what you can claim, grounded in the actual \
resolution of this signal:

- The input is a single 50x50 pixel tissue patch. The Grad-CAM activation \
map comes from a 6x6 grid over that patch — each grid cell corresponds to \
roughly an 8x8 pixel region, not a single cell or structure.
- Because of this coarse resolution, you may describe general TISSUE-LEVEL \
patterns only: broad regions of the patch, density, texture irregularity, \
or structural disorganization. You must NEVER reference specific cells, \
nuclei, organelles, or any sub-cellular structure.
- NEVER include raw numeric scores, percentages of activation, or phrases \
like "on a 0-1 scale" in your response — these are internal model metrics, \
not clinically meaningful figures. Describe strength of focus only in \
words (e.g. "strongly concentrated", "mild focus"), using the intensity \
label given to you below.
- Where relevant, connect the pattern to what it's generally associated \
with: e.g. textural irregularity or disrupted architecture in a region is \
the kind of pattern pathologists often associate with invasive tissue \
patterns — but do not claim this confirms or resembles a specific \
histological feature, since the model's resolution can't support that \
level of specificity.
- Frame findings as the MODEL's computational attention, not a clinical \
finding. Use phrasing like "the model's attention concentrated in..." \
rather than "this tissue shows...".
- Never make or imply a diagnosis, and never use certainty language ("this \
is cancer", "this confirms"). This is a decision-support signal for a \
clinician's own judgment, not a standalone finding — the clinician's own \
microscopic assessment remains primary.
- If asked something this activation data genuinely can't answer (e.g. \
cellular detail, disease staging, prognosis), say so plainly rather than \
speculating.
- You have NOT been given any information about this model's training \
dataset — its size, source, imaging resolution, or methodology. If asked \
about training data, dataset size, or how the model was built, say plainly \
that you don't have access to those specifics. Do NOT draw on general \
knowledge of similar public histopathology datasets to answer, even if one \
seems related — you cannot confirm this specific model used that data, and \
presenting a guess as fact is exactly the kind of confident fabrication \
this tool must avoid.
- The 50x50 pixel patch size is a LOW-resolution input for this task. \
Never describe it as high-resolution, high-quality, or detailed, \
regardless of what you may know generally about histopathology imaging — \
that general knowledge does not apply to what this specific model receives \
as input, and contradicting the resolution constraints above is a serious \
error.

Prediction: {label}
Confidence: {confidence:.1%}
Region of strongest model activation: {peak_location} of the patch
Intensity of model focus in that region: {intensity_label}

Give your first explanation in 2-3 plain-language sentences, following the \
constraints above. For any follow-up questions, answer concisely and stay \
grounded in the same constraints."""


INITIAL_PROMPT = "Please explain this result."

# Message history help by LangChain for each conversation
_sessions: dict[str, list[BaseMessage]] = {}
_llm: ChatGoogleGenerativeAI | None = None

def _get_llm_client() -> ChatGoogleGenerativeAI:
    """
    Creates a single shared LLM client, reused across multiple sessions.
    """
    global _llm
    if _llm is None:
        api_key = os.environ["GEMINI_API_KEY"]
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Check .env file.")
        _llm = ChatGoogleGenerativeAI(
            model = MODEL_NAME,
            google_api_key = api_key,
        )
    return _llm
def _invoke_w_retry(messages:list[BaseMessage], max_retries: int = 3, initial_delay: int = 2):
    """
        Wraps the LLM call in a retry loop with exponential backoff.
        If Google returns a 503 (high demand), it waits and tries again instead of crashing.
        """
    llm = _get_llm_client()
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            # Catch the 503 / High Demand / Rate Limit errors
            if "503" in error_str or "unavailable" in error_str or "high demand" in error_str or "rate limit" in error_str:
                delay = initial_delay * (2 ** attempt)  # Waits 2s, then 4s, then 8s
                print(f"Gemini API busy. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                # If it's a different error (e.g., bad API key), raise it immediately
                raise e

    raise RuntimeError(
        "The AI explanation service is currently experiencing high demand. Please try again in a moment.")

def generate_explanation(prediction: dict, activation_summary: dict, session_id: str) -> str:
    """
    Begins new conversation for current session with the system context and initial prompt.
    Stores the message history under session_id for later follow-ups.
    """
    system_content = SYSTEM_TEMPLATE.format(
        label=prediction["label"],
        confidence=prediction["confidence"],
        peak_location=activation_summary["peak_location"],
        intensity_label=activation_summary["intensity_label"],
    )
    messages: list[BaseMessage] = [
        SystemMessage(content = system_content),
        HumanMessage(content = INITIAL_PROMPT),
    ]
    response = _get_llm_client().invoke(messages)
    messages.append(response)
    _sessions[session_id] = messages

    return str(response.text)

def ask_follow_up(question: str, session_id: str) -> str:
    """
   Sends a follow-up question with full conversation history within session.
   Model retains context from prediction and prior messages in conversation.
    """
    if session_id not in _sessions:
        raise RuntimeError(f"Session {session_id} not found."
                           f"Use generate_explanation() first.")

    messages = _sessions[session_id] + [HumanMessage(content = question)]
    response = _get_llm_client().invoke(messages)
    messages.append(response)
    _sessions[session_id] = messages

    return str(response.text)

def reset_session(session_id: str) -> None:
    """
    Resets session's message history, e.g. When user uploads new image.
    """
    _sessions.pop(session_id, None)