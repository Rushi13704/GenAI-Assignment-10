
import logging
import os
import re
import time

import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("experiment10")


def search_web(topic: str) -> str:
    """Search the web for recent information about a topic."""
    results = DDGS().text(
        f"Latest information and important developments about {topic}",
        max_results=5,
    )

    if not results:
        return ""

    return "\n\n".join(
        f"Title: {item.get('title', '')}\n"
        f"Details: {item.get('body', '')}\n"
        f"URL: {item.get('href', '')}"
        for item in results
    )


def summarize_results(topic: str, search_results: str, llm=None) -> str:
    """Summarize search results into exactly three bullet points."""
    if llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Configure it in .env "
                "or your hosting platform's secrets."
            )

        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0.3,
            google_api_key=api_key,
        )


    prompt = f"""
You are a research assistant.

Topic: {topic}

Web search results:
{search_results}

Instructions:
1. Summarize the most important findings.
2. Return EXACTLY three bullet points.
3. Keep each bullet concise, informative, and factual.
4. Focus on relevant developments, facts, and implications.
5. Do not invent facts absent from the search results.
6. Mention uncertainty if the results are insufficient.
7. Output only the three bullet points.
"""

    response = llm.invoke(prompt)
    return response.content



def research_workflow(topic: str, search_fn=None, llm=None) -> str:
    """Search for a topic and create a three-point research brief."""
    topic = topic.strip()

    if not topic:
        raise ValueError("Please enter a research topic.")

    if search_fn is None:
        search_fn = search_web

    logger.info("Research request received: %s", topic)
    total_start = time.perf_counter()

    try:
        # Stage 1: Web search
        search_start = time.perf_counter()
        logger.info("Stage 1 started: DuckDuckGo search")

        results = search_fn(topic)

        search_seconds = time.perf_counter() - search_start
        logger.info("Stage 1 completed in %.2f seconds", search_seconds)

        if not results or not results.strip():
            raise RuntimeError("The web search returned no results.")

        # Stage 2: Gemini summarization
        summary_start = time.perf_counter()
        logger.info("Stage 2 started: Gemini summarization")

        brief = summarize_results(topic, results, llm=llm)

        summary_seconds = time.perf_counter() - summary_start
        logger.info("Stage 2 completed in %.2f seconds", summary_seconds)

        if not isinstance(brief, str) or not brief.strip():
            raise RuntimeError("The model returned an empty summary.")

        logger.info(
            "Total request time: %.2f seconds "
            "(search: %.2f seconds, Gemini: %.2f seconds)",
            time.perf_counter() - total_start,
            search_seconds,
            summary_seconds,
        )

        return brief

    except Exception:
        logger.exception(
            "Research request failed after %.2f seconds",
            time.perf_counter() - total_start,
        )
        raise



def main():
    st.set_page_config(
        page_title="Research Agent | Experiment 10",
        page_icon="🔎",
        layout="centered",
    )

    st.title("Research Task-Automation Agent")
    st.write(
        "Enter a topic to search the web and generate a concise "
        "three-point research brief."
    )

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Renewable energy developments in India",
    )

    if st.button("Research topic", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic first.")
            return

        with st.spinner("Searching the web and preparing your brief..."):
            try:
                brief = research_workflow(topic)
                st.subheader("Research Brief")

                bullets = [
                    line.strip()
                    for line in brief.splitlines()
                    if re.match(r"^\s*[-*•]\s+", line)
                ]

                if len(bullets) == 3:
                    for bullet in bullets:
                        st.markdown(bullet)
                else:
                    st.warning(
                        "The model did not return exactly three "
                        "bullet points. Showing its response below."
                    )
                    st.markdown(brief)

            except Exception as error:
                logger.error("Application request error: %s", error)
                st.error(
                    "The research request failed. Check the application "
                    "logs and API/search configuration, then try again."
                )


if __name__ == "__main__":
    main()
