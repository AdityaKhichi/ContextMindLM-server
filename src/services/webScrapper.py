from firecrawl import Firecrawl
from src.config.index import appConfig

firecrawl_client = Firecrawl(api_key=appConfig["firecrawl_api_key"])