import os
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_client() -> Optional[AzureOpenAI]:
    """Initialize Azure OpenAI client with error handling"""
    try:
        required_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return None
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",  # Updated to newer version
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        return None

client = initialize_client()
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def get_career_advice(resume_data: Dict[Any, Any]) -> str:
    """Generate in-depth, scored career analysis with improvement suggestions"""
    if not client:
        return "❌ Azure OpenAI service is not available. Please check your configuration."
    if not resume_data:
        return "❌ No resume data provided for analysis."
    try:
        # Refined prompt with scoring across key dimensions and improvement advice
        prompt = f"""
You are a seasoned career consultant. Analyze the following structured resume data and produce a detailed evaluation that includes:

**Resume Data:**
{resume_data}

Your response must contain the following sections, clearly labeled as Markdown headings:

1. **Design & Styling Assessment**
   - Evaluate the visual layout, formatting consistency, and readability.
   - Score (0–20 points).

2. **Skills & Competencies Evaluation**
   - Assess relevance and depth of listed skills. Identify top strengths and missing skills.
   - Score (0–20 points).

3. **Course & Learning Activities Review**
   - Review any courses, certifications, or training listed. Check relevance and completeness.
   - Score (0–20 points).

4. **Internship & Experience Analysis**
   - Examine internships and work experiences for impact and relevance.
   - Score (0–20 points).

5. **Overall Resume Score**
   - Provide a total score out of 100 by summing above categories.

6. **Improvement Recommendations**
   - For each category (Design, Skills, Courses, Internships), propose 2–3 actionable steps to improve the score.
   - Include specific examples (e.g., redesign suggestions, courses to add, internship strategies).

7. **Quick Summary Table**
   - Present a Markdown table with categories, scores, and top recommendation per category.

Format your analysis in concise, professional language. Use bullet points, tables, and clear headings. Deliver a rigorous, tailored assessment rather than general advice.
        """
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert career consultant specializing in resume evaluation with a quantitative scoring system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1200,
            top_p=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating career advice: {e}")
        return f"❌ Error generating career advice: {str(e)}"
