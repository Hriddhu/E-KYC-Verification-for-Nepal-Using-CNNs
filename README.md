# E-KYC

E-KYC is our minor project based on automated Know Your Customer verification for Nepali citizenship documents. The main idea of this project is to make the KYC process faster and more reliable by using computer vision and deep learning.

Instead of checking citizenship documents manually, this system tries to verify the document and the user through different modules like OCR, face matching, liveness detection, stamp verification, and document tampering detection.

This project is mainly built for academic and demonstration purposes, but the idea is inspired by real problems faced in online identity verification.

---

## About the Project

In many KYC systems, citizenship documents are still checked by people manually. This takes time, and sometimes small changes in documents can be missed. Fake documents, edited photos, wrong details, and spoofed face images are also major problems in online verification.

Our project tries to handle these problems by checking different parts of the KYC process automatically.

The system takes citizenship images and a selfie from the user. Then it checks the document fields, reads the text, compares the face, checks if the selfie is real, verifies the stamp, and also checks if the document looks tampered or genuine.

At the end, the system gives a final KYC result based on all these checks.

---

## What the System Checks

The system checks multiple things instead of depending on only one model.

It checks:

- whether important citizenship fields are detected properly
- whether the extracted text matches the user-entered details
- whether the document photo and selfie belong to the same person
- whether the selfie is from a real person or a spoof attempt
- whether the stamp/logo is similar to the original reference
- whether the citizenship document looks genuine or tampered

This makes the final decision more reliable because one weak result does not directly decide everything.

---

## Main Modules

### 1. Document Field Detection

The document detection part is used to find important regions from the citizenship image. These regions are cropped and sent to other modules like OCR, face matching, and stamp checking.

Some of the detected fields include:

- citizenship photo
- full name
- date of birth
- citizenship number
- permanent address
- stamp/logo
- other useful citizenship regions

This helps the system focus only on the important parts of the document instead of processing the whole image blindly.

---

### 2. OCR and Field Matching

After the fields are detected, OCR is used to extract text from the document.

The extracted text is compared with the details entered by the user, such as name, date of birth, citizenship number, and address.

This helps the system check whether the information provided by the user matches the information found in the citizenship document.

---

### 3. Face Matching

The face matching module compares the face from the citizenship document with the selfie uploaded by the user.

This is important because even if the document is real, someone else could try to use it for verification. Face matching helps check whether the person submitting the KYC is actually the same person shown in the document.

---

### 4. Passive Liveness Detection

The liveness module checks whether the selfie is from a real person or not.

This helps protect the system from spoofing attacks like:

- printed photos
- mobile screen images
- replay attacks
- fake face submissions

This module adds another layer of security to the KYC process.

---

### 5. Stamp Verification

The system also checks the government stamp or logo found in the citizenship document.

It compares the detected stamp with stored reference templates. This is useful because fake or edited documents may have a wrong, unclear, or manipulated stamp.

---

### 6. Document Tampering Detection

The tampering detection module checks whether the citizenship image looks genuine or edited.

This module is important because many forged documents may look normal at first glance, but small parts like the name, date of birth, citizenship number, address, or photo may have been changed.

Some examples of tampering are:

- edited name
- changed date of birth
- modified citizenship number
- replaced photo
- edited address
- inserted text
- overwritten document regions

---

## Final KYC Decision

The final decision is not made from only one module.

The system combines results from:

- OCR matching
- face similarity
- liveness detection
- stamp verification
- tampering detection

After combining these results, the system generates a final score and gives the KYC result as approved or rejected.

This approach is better than depending on only one model because KYC verification has many different risks.

---

## Basic Workflow

The project workflow is:

1. User fills the KYC form.
2. User uploads citizenship front image.
3. User uploads citizenship back image.
4. User uploads a selfie image.
5. The system detects important document regions.
6. OCR extracts text from the detected fields.
7. Extracted text is compared with the form data.
8. The citizenship photo is compared with the selfie.
9. Liveness detection checks whether the selfie is real.
10. Stamp verification checks the document stamp/logo.
11. Tampering detection checks whether the document is edited.
12. The system gives the final KYC result.

---

## Tech Stack

### Frontend

- React
- Vite
- Tailwind CSS
- JavaScript
- React Router

### Backend

- FastAPI
- Uvicorn
- Python Multipart
- CORS Middleware

### Machine Learning and Computer Vision

- Python
- OpenCV
- NumPy
- PyTorch
- Torchvision
- timm
- YOLO / Ultralytics
- EasyOCR
- InsightFace
- ONNX Runtime
- MediaPipe
- Albumentations
- scikit-learn
- scikit-image

---

## Project Structure

```bash
E-KYC/
│
├── backend/
│   └── liveness_api.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── eslint.config.js
│
├── ml/
│   ├── docservice/
│   ├── face-service/
│   ├── liveness-service/
│   ├── ocr-service/
│   └── tampering/
│
├── requirements.txt
├── README.md
└── .gitignore
