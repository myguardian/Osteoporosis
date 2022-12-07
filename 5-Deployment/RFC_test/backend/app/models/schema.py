from pydantic import BaseModel

class PatientInfo(BaseModel):
    age: float
    sex: int
    weight: float
    height: float
    parentbreak: int
    alcohol: int
    obreak: int
    arthritis: int
    diabetes: int
    oralster: int
    smoke: int