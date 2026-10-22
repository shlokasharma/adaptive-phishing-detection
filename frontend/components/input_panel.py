from __future__ import annotations

import streamlit as st


def render_input_panel() -> dict:
    """
    Render the user input interface.

    Returns
    -------
    dict
        Structured input containing the selected analysis type
        and the corresponding user-provided information.
    """

    st.markdown(
        '<div class="section-heading">Analysis Input</div>',
        unsafe_allow_html=True,
    )

    analysis_type = st.radio(
        "Choose what you want to analyze",
        options=[
            "Email",
            "URL",
            "Combined",
        ],
        horizontal=True,
        key="analysis_type",
    )

    input_data = {
        "analysis_type": analysis_type,
        "subject": "",
        "body": "",
        "sender": "",
        "receiver": "",
        "url": "",
        "submitted": False,
    }

    # ------------------------------------------------------------------
    # Email analysis
    # ------------------------------------------------------------------

    if analysis_type == "Email":

        sender = st.text_input(
            "Sender",
            placeholder="sender@example.com",
        )

        receiver = st.text_input(
            "Receiver",
            placeholder="recipient@example.com",
        )

        subject = st.text_input(
            "Email Subject",
            placeholder="Enter the email subject...",
        )

        body = st.text_area(
            "Email Body",
            placeholder=(
                "Paste the complete email content here..."
            ),
            height=260,
        )

        input_data["sender"] = sender
        input_data["receiver"] = receiver
        input_data["subject"] = subject
        input_data["body"] = body

    # ------------------------------------------------------------------
    # URL analysis
    # ------------------------------------------------------------------

    elif analysis_type == "URL":

        url = st.text_input(
            "URL",
            placeholder="https://example.com",
        )

        input_data["url"] = url

    # ------------------------------------------------------------------
    # Combined analysis
    # ------------------------------------------------------------------

    elif analysis_type == "Combined":

        left_column, right_column = st.columns(2)

        with left_column:

            sender = st.text_input(
                "Sender",
                placeholder="sender@example.com",
            )

            receiver = st.text_input(
                "Receiver",
                placeholder="recipient@example.com",
            )

            subject = st.text_input(
                "Email Subject",
                placeholder="Enter the email subject...",
            )

        with right_column:

            url = st.text_input(
                "URL",
                placeholder="https://example.com",
            )

        body = st.text_area(
            "Email Body",
            placeholder="Paste the complete email content here...",
            height=220,
        )

        input_data["sender"] = sender
        input_data["receiver"] = receiver
        input_data["subject"] = subject
        input_data["body"] = body
        input_data["url"] = url

    st.markdown("")

    submitted = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True,
    )

    input_data["submitted"] = submitted

    return input_data