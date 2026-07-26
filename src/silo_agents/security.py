from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Classification, Domain

_CLASSIFICATION_ORDER = {
    Classification.PUBLIC: 0,
    Classification.INTERNAL: 1,
    Classification.RESTRICTED: 2,
}


class RetrievalPrincipal(BaseModel):
    """Service identity used to authorize retrieval before vector search."""

    principal_id: str
    allowed_domains: set[Domain]
    max_classification: Classification = Classification.INTERNAL
    allowed_record_ids: set[str] | None = Field(default=None)

    def assert_domain(self, domain: Domain) -> None:
        if domain not in self.allowed_domains:
            raise PermissionError(
                f"Principal {self.principal_id!r} is not allowed to search {domain.value!r}"
            )

    def allows(self, domain: Domain, classification: Classification, record_id: str) -> bool:
        if domain not in self.allowed_domains:
            return False
        if _CLASSIFICATION_ORDER[classification] > _CLASSIFICATION_ORDER[self.max_classification]:
            return False
        return self.allowed_record_ids is None or record_id in self.allowed_record_ids

    def allowed_classifications(self) -> list[Classification]:
        threshold = _CLASSIFICATION_ORDER[self.max_classification]
        return [value for value, rank in _CLASSIFICATION_ORDER.items() if rank <= threshold]
