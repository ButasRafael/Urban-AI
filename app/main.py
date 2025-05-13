from fastapi import FastAPI
from starlette_exporter import PrometheusMiddleware, handle_metrics
import logging
from pythonjsonlogger import jsonlogger
from app.api import auth
from app.api.inference_routes import router as infer_router
from app.api import problems, analytics
from app.core.database import Base, engine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.tracing import Transaction 
from fastapi import Request


from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import os
import logging
from app.core.security import require_roles

logger = logging.getLogger(__name__)

sentry_logging = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR, 
)

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FastApiIntegration(),
        SqlalchemyIntegration(),
        sentry_logging,
    ],
    traces_sample_rate=1.0,  
    environment=os.getenv("ENVIRONMENT", "production"),
    release=os.getenv("RELEASE"),
    send_default_pii=True,
    profile_session_sample_rate=1.0,            
)


handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s'
)
handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

resource = Resource.create({
    "service.name": "urban-ai-api",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)

otlp_exporter = OTLPSpanExporter(endpoint="http://tempo:4318/v1/traces")

processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(processor)

console_exporter = ConsoleSpanExporter()
provider.add_span_processor(SimpleSpanProcessor(console_exporter))

trace.set_tracer_provider(provider)


app = FastAPI(title="Urban AI API")
FastAPIInstrumentor.instrument_app(app, tracer_provider=provider) 

LoggingInstrumentor().instrument(set_logging_format=True)

RequestsInstrumentor().instrument() 

app.add_middleware(SentryAsgiMiddleware) 
app.add_middleware(PrometheusMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/healthz",dependencies=[require_roles("admin")])
def _ping():
    return {"status": "ok"}

@app.get("/sentry-test",dependencies=[require_roles("admin")])
def test():
    try:
        1/0
    except Exception as e:
        logger.error("💥 test crash", exc_info=e)
        return {"ok": False}


app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(infer_router)
app.include_router(problems.router)
app.include_router(analytics.router) 

app.add_route(
    "/metrics/raw",
    handle_metrics,
    methods=["GET"],
    include_in_schema=False, 
)


@app.get("/metrics", dependencies=[require_roles("admin")])
async def metrics(request: Request):
    """
    Return the metrics in a format that can be scraped by Prometheus.
    """
    return handle_metrics(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)