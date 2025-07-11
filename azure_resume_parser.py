import os
import re
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_TEXT_KEY = os.getenv("AZURE_TEXT_KEY")
AZURE_TEXT_ENDPOINT = os.getenv("AZURE_TEXT_ENDPOINT")

# Utility functions for email and phone extraction
def extract_email(text):
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group() if match else None

def extract_phone(text):
    match = re.search(r'\+?\d[\d\s\-()]{7,}\d', text)
    return match.group() if match else None

def extract_resume_data_full(pdf_path: str) -> dict:
    # Clients
    form_client = DocumentAnalysisClient(
        endpoint=AZURE_FORM_ENDPOINT,
        credential=AzureKeyCredential(AZURE_FORM_KEY)
    )
    text_client = TextAnalyticsClient(
        endpoint=AZURE_TEXT_ENDPOINT,
        credential=AzureKeyCredential(AZURE_TEXT_KEY)
    )

    with open(pdf_path, "rb") as f:
        poller = form_client.begin_analyze_document("prebuilt-document", document=f)
    result = poller.result()

    # Full text for regex and NER
    full_text = " ".join([p.content for p in result.paragraphs])

    # Initial data structure
    data = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "projects": [],
        "education": [],
        "experience": [],
        "certifications": [],
        "others": []
    }

    # Key-Value Pair Extraction (quick wins)
    for kv in result.key_value_pairs:
        if kv.key and kv.value:
            key = kv.key.content.lower()
            val = kv.value.content
            if "name" in key and not data["name"]: data["name"] = val
            elif "email" in key and not data["email"]: data["email"] = val
            elif "phone" in key and not data["phone"]: data["phone"] = val

    # Backup: Regex for email and phone
    if not data["email"]:
        data["email"] = extract_email(full_text)
    if not data["phone"]:
        data["phone"] = extract_phone(full_text)

    # NER for Name if still missing
    if not data["name"]:
        response = text_client.recognize_entities([full_text])
        for doc in response:
            if not doc.is_error:
                for entity in doc.entities:
                    if entity.category == "Person":
                        data["name"] = entity.text
                        break

    # Paragraph Scanning for Skills, Projects, etc.
    for para in result.paragraphs:
        content = para.content.lower()

        # Skills detection
        if any(word in content for word in ["skill", "technologies", "tools"]):
            data["skills"].append(para.content)

        # Project detection
        if any(word in content for word in ["project", "developed", "built", "designed"]):
            data["projects"].append(para.content)

        # Education detection
        if any(word in content for word in ["education", "bachelor", "master", "degree", "university", "college"]):
            data["education"].append(para.content)

        # Experience detection
        if any(word in content for word in ["experience", "worked", "internship", "employment", "job"]):
            data["experience"].append(para.content)

        # Certifications detection
        if "certification" in content or "certificate" in content:
            data["certifications"].append(para.content)

        # Others
        if not any(word in content for word in ["skill", "project", "education", "experience", "certification"]):
            data["others"].append(para.content)

    # Optional: Clean duplicates, strip text
    for key in ["skills", "projects", "education", "experience", "certifications", "others"]:
        data[key] = list(set([x.strip() for x in data[key]]))

    return data


# Example usage:
pdf_path = "C:\Projects\careerqr\sample_resume.pdf"
resume_info = extract_resume_data_full(pdf_path)
print(resume_info)
