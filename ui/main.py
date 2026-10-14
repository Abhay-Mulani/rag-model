import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "300"))


def send_feedback(query: str, rating: int, session_id: str) -> None:
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={"query": query, "rating": rating, "session_id": session_id},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        st.error(f"Unable to save feedback: {error}")
    else:
        st.toast("Thanks for the feedback.")


st.set_page_config(page_title="AlphaFin Market Intelligence", page_icon="📈")
st.title("AlphaFin Market Intelligence")
st.caption(
    "Educational market intelligence grounded in cited research. "
    "Not financial advice."
)

risk_appetite = st.selectbox("Risk appetite", ["conservative", "moderate", "aggressive"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("confidence") is not None:
            st.progress(message["confidence"], text=f"Confidence: {message['confidence']:.0%}")

if prompt := st.chat_input("How can we help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": prompt, "risk_appetite": risk_appetite},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        st.error(f"Unable to contact the API: {error}")
    else:
        with st.chat_message("assistant"):
            st.markdown(payload["answer"])
            metric_columns = st.columns(3)
            metric_columns[0].metric("Intent", payload["intent"].replace("_", " ").title())
            metric_columns[1].metric("Confidence", f"{payload['confidence']:.0%}")
            metric_columns[2].metric("Sources", len(payload["sources"]))
            st.progress(
                payload["confidence"],
                text=f"Validation confidence: {payload['confidence']:.0%}",
            )

            st.subheader("Evidence and sources")
            for source in payload["sources"]:
                st.write(f"- `{source}`")
            with st.expander("Retrieved evidence"):
                for index, context in enumerate(payload["retrieved_context"], start=1):
                    st.markdown(f"**Evidence {index}**")
                    st.write(context)

            with st.expander("Agent workflow"):
                st.write(" → ".join(payload["agents_invoked"]))
                st.caption(
                    "Planner identifies intent, Retriever finds evidence, a "
                    "specialist prepares the response, and Validator checks grounding."
                )

            if payload["risk_assessment"]:
                with st.expander("Risk assessment"):
                    st.write(payload["risk_assessment"])
            if payload["analysis"]:
                with st.expander("Analysis output"):
                    st.write(payload["analysis"])

            feedback_columns = st.columns(2)
            if feedback_columns[0].button("👍 Helpful", key=f"helpful-{payload['session_id']}"):
                send_feedback(prompt, 1, payload["session_id"])
            if feedback_columns[1].button("👎 Needs improvement", key=f"needs-work-{payload['session_id']}"):
                send_feedback(prompt, -1, payload["session_id"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": payload["answer"],
                "sources": payload["sources"],
                "agents": payload["agents_invoked"],
                "confidence": payload["confidence"],
            }
        )
