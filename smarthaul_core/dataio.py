from pydantic import BaseModel, Field, ValidationError, field_validator
import pandas as pd
import streamlit as st

REQUIRED_COLS = ["order_id","lat","lon","demand","window_start_min","window_end_min"]

class Order(BaseModel):
    order_id: str
    lat: float
    lon: float
    demand: int = Field(ge=0)
    window_start_min: int = Field(ge=0, le=24*60)
    window_end_min: int = Field(ge=0, le=24*60)

    @field_validator("window_end_min")
    def check_window(cls, v, info):
        if v < info.data["window_start_min"]:
            raise ValueError("window_end_min < window_start_min")
        return v

@st.cache_data(ttl=900, show_spinner=False)
def load_and_validate_orders(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    errors, good = [], []
    for i, row in df.iterrows():
        try:
            good.append(Order(**row.to_dict()).model_dump())
        except ValidationError as e:
            errs = [f"{x['loc']} -> {x['msg']}" for x in e.errors()]
            errors.append(f"Row {i}: {', '.join(errs)}")
    if errors:
        raise ValueError("Invalid rows:\n" + "\n".join(errors[:20]))
    return pd.DataFrame(good)
