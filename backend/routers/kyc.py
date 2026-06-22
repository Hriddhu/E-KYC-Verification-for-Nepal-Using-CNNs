from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request

from backend.schemas.kyc import KYCFormData, KYCResponse

router = APIRouter(prefix="/api/kyc", tags=["KYC"])


@router.post("/upload", response_model=KYCResponse)
async def upload_kyc_documents(
    request: Request,
    full_name: str = Form(...),
    date_of_birth: str = Form(...),
    gender: str = Form(...),
    citizenship_number: str = Form(...),
    permanent_address: str = Form(...),
    current_address: str = Form(...),
    selfie_image: UploadFile = File(...),
    document_front: UploadFile = File(...),
    document_back: UploadFile = File(...),
):
    # 1. build KYCFormData from raw form fields (triggers validation)


    # 2. grab preloaded models from request.app.state


    # 3. call services in order:
    #    - document detection (front + back)
    #    - ocr + accuracy check
    #    - face similarity
    #    - stamp similarity
    #    - passive liveness
    #    - document tampering


    # 4. call decision service to compute final verdict


    # 5. return KYCResponse
    pass