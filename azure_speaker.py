import os
import logging
import asyncio
import html  # Import the html module for escaping special characters
from typing import Optional, Dict, Any, Union
from enum import Enum
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class VoiceType(Enum):
    """Available neural voices"""
    JENNY = "en-US-JennyNeural"  # Female, friendly
    DAVIS = "en-US-DavisNeural"  # Male, warm
    ARIA = "en-US-AriaNeural"  # Female, cheerful
    GUY = "en-US-GuyNeural"  # Male, warm
    JANE = "en-US-JaneNeural"  # Female, confident
    JASON = "en-US-JasonNeural"  # Male, casual
    SARA = "en-US-SaraNeural"  # Female, cheerful
    TONY = "en-US-TonyNeural"  # Male, authority


class SpeechRate(Enum):
    """Speech rate options"""
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    X_SLOW = "x-slow"
    X_FAST = "x-fast"


class AzureSpeechService:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")
        self.speech_config = None
        self._initialize_config()

    def _initialize_config(self) -> None:
        """Initialize Azure Speech configuration"""
        try:
            if not self.speech_key or not self.speech_region:
                logger.error("Missing Azure Speech Service credentials (AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)")
                return

            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region
            )
            # Set a default voice
            self.speech_config.speech_synthesis_voice_name = VoiceType.JENNY.value
            logger.info("Azure Speech Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech Service: {e}")

    def _create_ssml(self, text: str, voice: VoiceType, rate: SpeechRate) -> str:
        """Create SSML (Speech Synthesis Markup Language) for better control."""
        # Escape special XML characters in the text to prevent synthesis errors
        escaped_text = html.escape(text)
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice.value}">
                <prosody rate="{rate.value}">
                    {escaped_text}
                </prosody>
            </voice>
        </speak>
        """

    def speak_text(self, text: str, voice: VoiceType = VoiceType.JENNY,
                   rate: SpeechRate = SpeechRate.MEDIUM,
                   save_to_file: Optional[str] = None) -> bool:
        """
        [SERVER-SIDE] Convert text to speech and play on the server's default audio output.

        Args:
            text: Text to convert to speech.
            voice: Voice type to use.
            rate: Speech rate.
            save_to_file: Optional path to save audio file instead of playing.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.speech_config:
            logger.error("Azure Speech Service not initialized")
            return False
        if not text or not text.strip():
            logger.warning("Empty text provided for speech synthesis")
            return False

        try:
            ssml_text = self._create_ssml(text, voice, rate)
            audio_config = speechsdk.audio.AudioOutputConfig(filename=save_to_file) if save_to_file else None
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Speech synthesis completed successfully.")
                return True
            else:
                cancellation_details = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.error_details:
                    logger.error(f"Error details: {cancellation_details.error_details}")
                return False
        except Exception as e:
            logger.error(f"Error during speech synthesis: {e}")
            return False

    def synthesize_speech_to_memory(self, text: str, voice: VoiceType = VoiceType.JENNY,
                                    rate: SpeechRate = SpeechRate.MEDIUM) -> Optional[bytes]:
        """
        [CLIENT-SIDE SUPPORT] Synthesizes text to speech and returns the audio data as bytes.
        This is ideal for sending audio to a web client like Streamlit.

        Args:
            text: Text to convert to speech.
            voice: Voice type to use.
            rate: Speech rate.

        Returns:
            Optional[bytes]: The WAV audio data, or None if an error occurred.
        """
        if not self.speech_config:
            logger.error("Azure Speech Service not initialized")
            return None
        if not text or not text.strip():
            logger.warning("Empty text provided for speech synthesis")
            return None

        try:
            ssml_text = self._create_ssml(text, voice, rate)
            # Synthesize to an in-memory stream by setting audio_config to None
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Speech synthesis to memory completed successfully.")
                return result.audio_data
            else:
                cancellation_details = result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.error_details:
                    logger.error(f"Error details: {cancellation_details.error_details}")
                return None
        except Exception as e:
            logger.error(f"Error during in-memory speech synthesis: {e}")
            return None


# --- Global Instance and Wrapper Functions ---

speech_service = AzureSpeechService()

# Mapping dictionaries for wrapper functions
VOICE_MAPPING = {
    "jenny": VoiceType.JENNY, "davis": VoiceType.DAVIS, "aria": VoiceType.ARIA,
    "guy": VoiceType.GUY, "jane": VoiceType.JANE, "jason": VoiceType.JASON,
    "sara": VoiceType.SARA, "tony": VoiceType.TONY
}
RATE_MAPPING = {
    "slow": SpeechRate.SLOW, "medium": SpeechRate.MEDIUM, "fast": SpeechRate.FAST,
    "x-slow": SpeechRate.X_SLOW, "x-fast": SpeechRate.X_FAST
}


def get_speech_audio_data(text: str, voice: str = "jenny", rate: str = "medium") -> Optional[bytes]:
    """
    [NEW] Wrapper function to get synthesized speech as audio data bytes for client-side playback.
    This is the function your Streamlit app should use with `st.audio`.

    Args:
        text: Text to speak.
        voice: Voice name (jenny, davis, aria, etc.).
        rate: Speech rate (slow, medium, fast).

    Returns:
        Audio data in bytes, or None on failure.
    """
    try:
        selected_voice = VOICE_MAPPING.get(voice.lower(), VoiceType.JENNY)
        selected_rate = RATE_MAPPING.get(rate.lower(), SpeechRate.MEDIUM)
        return speech_service.synthesize_speech_to_memory(text, selected_voice, selected_rate)
    except Exception as e:
        logger.error(f"Error in get_speech_audio_data wrapper: {e}")
        return None


def speak_text(text: str, voice: str = "jenny", rate: str = "medium",
               save_to_file: Optional[str] = None) -> bool:
    """
    [Legacy] Simple wrapper function to play audio on the server or save to a file.

    Args:
        text: Text to speak.
        voice: Voice name (jenny, davis, aria, etc.).
        rate: Speech rate (slow, medium, fast).
        save_to_file: Optional path to save audio file.
    """
    try:
        selected_voice = VOICE_MAPPING.get(voice.lower(), VoiceType.JENNY)
        selected_rate = RATE_MAPPING.get(rate.lower(), SpeechRate.MEDIUM)
        return speech_service.speak_text(text, selected_voice, selected_rate, save_to_file)
    except Exception as e:
        logger.error(f"Error in speak_text wrapper: {e}")
        return False