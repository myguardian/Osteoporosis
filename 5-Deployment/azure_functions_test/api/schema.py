from pydantic import BaseModel

# schema for patient info
# id can be omitted for security
class PatientInfo(BaseModel):
    id: int
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