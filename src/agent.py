from __future__ import annotations

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)
    logger.info("agent started")


def get_instruction():
    system_prompt = "".join(
        [
            "You are a 22-year-old, final-year Japanese university student"
            + "who is currently looking for a job. ",
            "You have no prior work experience and are not used to interviews. ",
            "Currently, you are interviewing with a hiring manager from a company you applied to, ",
            "and your goal is to get an offer.",
            "\n\n",
            "The company you applied is an IT consulting company specialized in AI products. ",
            "The job position is an entry-level AI engineer. ",
            "In general, you are expected to answer questions related to your background, "
            + "research, hobbies, and job hunting status. ",
            "The interviewer might also ask you detailed technical and behavioral questions "
            + "to understand your thinking process and motivation. ",
            "\n\n",
            "Remember, you should behave as a student who is not used to interviews. ",
            "Craft your response so that it befits your age . ",
        ]
    )
    return system_prompt


def get_model(backend: Literal["OpenAI", "Azure"] = "OpenAI"):
    if backend == "OpenAI":
        model = openai.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview-2024-12-17",
            instructions=get_instruction(),
            modalities=["audio", "text"],
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500
            ),
            voice="sage",
            temperature=0.7,  # recommended >=0.6,
        )
    else:
        model = openai.realtime.RealtimeModel.with_azure(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            instructions=get_instruction(),
            modalities=["audio", "text"],
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500
            ),
            voice="sage",
            temperature=0.7,  # recommended >=0.6,
        )

    return model


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    agent = MultimodalAgent(model=get_model(backend="OpenAI"), chat_ctx=llm.ChatContext())
    agent.start(ctx.room, participant)
    agent.generate_reply()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
