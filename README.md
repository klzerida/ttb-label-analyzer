---
title: TTB Label Analyzer
emoji: 🍷
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

# TTB Label Analyzer

## Overview

TTB Label Analyzer is an AI-assisted prototype that reviews alcohol beverage labels for potential compliance issues based on U.S. Alcohol and Tobacco Tax and Trade Bureau (TTB) labeling guidance. It accepts a label image, extracts visible text using OCR, and uses a large language model to flag potentially missing or non-compliant label elements. The tool is intended to assist reviewers by highlighting possible compliance concerns, not to replace official TTB review or legal compliance verification. See TTB wine labeling guidance for background.

***

## Features

- Upload beverage label images.
- Extract text with optical character recognition (OCR).
- Run AI-assisted compliance analysis.
- Detect potentially missing required label information.
- Generate a structured compliance review report.
- Provide a web-based interface.

***

## Technology Stack

### Frontend
- Gradio

### OCR
- EasyOCR

### AI Model
- Qwen/Qwen2.5-72B-Instruct

### Inference
- Hugging Face Inference API

### Language
- Python 3.11+

***

## Workflow

### 1. Text Extraction

The uploaded label image is processed with EasyOCR to extract visible text.

### 2. Compliance Analysis

The extracted text is sent to the Qwen 2.5 72B Instruct model using a structured compliance-review prompt. The model evaluates common label elements such as brand name, class or type designation, alcohol content, net contents, health warning statements, and name/address information, consistent with TTB labeling checklists.

### 3. Report Generation

The model returns a report that includes compliance findings, missing elements, potential issues, and recommendations for further review.

***

## Assumptions

This prototype assumes:

- Labels are primarily written in English.
- Uploaded images are reasonably clear and readable.
- OCR quality directly affects analysis accuracy.
- AI-generated findings are advisory and require human review.

TTB guidance shows that mandatory label information varies by beverage type, so the analyzer should be treated as a screening tool rather than a final compliance authority.

***

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

***

## Setup Instructions

### Clone the Repository

```bash
git clone https://huggingface.co/spaces/klzerida/ttb-label-analyzer
cd ttb-label-analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Set a Hugging Face API token (with access to use Qwen/Qwen2.5-72B-Instruct via the Inference API):

```bash
export HF_TOKEN=your_token_here
```

On Windows (PowerShell):

```powershell
set HF_TOKEN=your_token_here
```

Do not share or commit your actual token; treat it like a password.

### Run the Application

```bash
python app.py
```

The application will launch locally and provide a Gradio URL.

***

## Deployment

### Live Application

Replace the placeholder with your deployed Space URL:

```markdown
[https://username-ttb-label-analyzer.hf.space](https://username-ttb-label-analyzer.hf.space)
```

***

## Limitations

- OCR errors may affect analysis quality.
- The app currently supports image-only review, not PDF parsing.
- The prototype focuses on common label requirements and does not cover every TTB rule.
- Results should be reviewed by qualified compliance personnel.

TTB labeling resources emphasize mandatory information and approval processes, which differ across wine, malt beverages, and distilled spirits; this tool should remain advisory.

***

## Future Enhancements

- PDF support
- Label layout analysis
- Confidence scoring
- Batch processing
- Automated COLA validation
- Fine-tuned compliance model

***

## Author

Yong Chi  

M.S. Electrical and Computer Engineering  
San Francisco State University