import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="config/.env", extra="ignore")

    ts01_username: str = Field(default_factory=lambda: os.getenv("USER", ""))

    models: list[str]
    inclusion_model: str
    inclusion_label: str
    embedding_model: str = "allenai/sclite-scite"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=".config/config.yaml"),
        )


settings = Settings()
