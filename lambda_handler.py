"""
Lambda entrypoint.
Wraps the existing Flask `application` object with Mangum so API Gateway
(HTTP API, payload format 2.0) can invoke it as a Lambda function.

"""
 
from mangum import Mangum
from application import application
 
# api_gateway_base_path=None works for both HTTP API (v2) and REST API (v1)
# lifespan="off" because Flask (WSGI) has no async lifespan events
handler = Mangum(application, lifespan="off")