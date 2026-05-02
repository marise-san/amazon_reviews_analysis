from pydantic import BaseModel, Field

class ReviewInput(BaseModel):
    review_text: str = Field(..., title="Texto da Avaliacao", description="Texto da avaliacao do produto a ser analisado.")

class PredictionOutput(BaseModel):
    prediction_id: int = Field(..., title="Classe Predita Interna", description="ID gerado pelo SparkML.")
    sentiment: str = Field(..., title="Sentimento", description="Positivo, Neutro ou Negativo.")
    confidence_percentage: str = Field(..., title="Grau de Confianca", description="Porcentagem de certeza do modelo.")
    human_readable_message: str = Field(..., title="Mensagem", description="Descricao textual do resultado.")
