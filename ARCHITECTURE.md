# ZXing Barcode Processing Architecture

## Overview

This document outlines the recommended architecture for a high-throughput barcode processing system capable of handling 1 million packages per day using ZXing-CPP and Azure cloud services.

## Business Requirements

- **Throughput**: 1,000,000 packages per day (11.6 requests/second)
- **Processing Time**: <1 second per image
- **Reliability**: 99%+ success rate
- **Scalability**: Auto-scale based on demand
- **Integration**: Support both HTTP API and direct queue access

## Architecture Overview

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐
│   HTTP API      │    │   Direct Queue   │
│   (Testing)     │    │   (Production)   │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌─────────────────┐
            │  Azure Service  │
            │  Bus Queue      │
            └─────────────────┘
                     │
            ┌─────────────────┐
            │  ZXing Workers  │
            │  (Auto-scaling) │
            └─────────────────┘
                     │
            ┌─────────────────┐
            │  Result Storage │
            │  (Cosmos DB)    │
            └─────────────────┘
```

## Core Components

### 1. HTTP API Server (Azure App Service)

**Purpose**: Handle HTTP requests for testing and external clients

**Technology**: FastAPI on Azure App Service

**Key Features**:
- Image upload handling
- Job ID generation
- Result polling endpoint
- Webhook callback support

**Implementation**:
```python
@app.post("/detect_barcodes")
async def detect_barcodes(file: UploadFile = File(...), callback_url: str = None):
    job_id = await submit_to_queue(file, callback_url)
    return {"jobId": job_id, "status": "queued", "pollUrl": f"/result/{job_id}"}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    result = await poll_output_queue(job_id)
    return result or {"status": "processing", "jobId": job_id}
```

### 2. Azure Service Bus

**Purpose**: Message queue for job distribution and result collection

**Configuration**:
- **Input Queue**: `zxing-input-queue`
- **Output Queue**: `zxing-output-queue`
- **Max Delivery Count**: 3 (retry failed jobs)
- **Dead Letter Queue**: Enabled
- **Message TTL**: 1 hour (input), 24 hours (output)

**Message Format**:
```json
{
  "jobId": "job_12345_20241201_143022",
  "imageBlobUrl": "https://storage.blob.core.windows.net/images/job_12345.jpg",
  "timestamp": "2024-12-01T14:30:22Z",
  "clientId": "client_abc123",
  "priority": "normal"
}
```

### 3. ZXing Processing Workers (Azure Container Instances)

**Purpose**: Pure barcode processing containers (no HTTP server)

**Technology**: Python containers with ZXing-CPP

**Key Features**:
- Competing consumers pattern
- Batch message processing
- Automatic error handling
- Health monitoring

**Implementation**:
```python
class ZXingWorker:
    async def process_job(self, message):
        try:
            # Download image from blob storage
            image_data = await self.download_image(message.image_blob_url)
            
            # Process with ZXing
            start_time = time.time()
            barcodes = self.model.detect_barcodes(image_data)
            processing_time = time.time() - start_time
            
            # Send result to output queue
            await self.send_result(message.job_id, barcodes, processing_time)
            
            # Complete message
            await receiver.complete_message(message)
            
        except Exception as e:
            # Let Service Bus handle retry logic
            logger.error(f"Error processing job {message.job_id}: {e}")
```

### 4. Azure Blob Storage

**Purpose**: Store uploaded images and processing results

**Configuration**:
- **Hot Tier**: First 30 days
- **Cool Tier**: Days 31-90
- **Archive Tier**: 90+ days

**Lifecycle Management**:
- Automatic tier transitions
- Cost optimization
- Retention policies

### 5. Azure Cosmos DB

**Purpose**: Store barcode processing results and metadata

**Configuration**:
- **Provisioned Throughput**: 1,000 RU/s
- **Partition Key**: job_id
- **TTL**: 90 days (active), then archive

**Document Structure**:
```json
{
  "id": "job_12345_20241201_143022",
  "jobId": "job_12345_20241201_143022",
  "status": "success",
  "barcodes": [
    {
      "value": "J18CBEP8CCN070812400095N",
      "format": "CODE128",
      "confidence": 0.95,
      "position": {"x": 100, "y": 200, "width": 300, "height": 50}
    }
  ],
  "processingTime": 0.86,
  "timestamp": "2024-12-01T14:30:23Z",
  "workerId": "worker_001"
}
```

## Performance Characteristics

### Single Container Performance
- **Processing Time**: ~0.86 seconds per image
- **Throughput**: 1.23 requests/second
- **Memory Usage**: ~121MB per container
- **CPU Usage**: ~0.2% per container

### Scaling Requirements
- **Target Throughput**: 11.6 requests/second
- **Required Containers**: 10 containers (linear scaling)
- **Buffer Capacity**: 12 containers (20% headroom)
- **Expected Performance**: 14.8 requests/second

## Integration Options

### 1. HTTP API (Testing/External)
**Use Case**: Testing, external clients, web interfaces

**Pros**:
- Simple integration
- Standard HTTP protocol
- Easy testing and debugging

**Cons**:
- Higher latency
- More resource intensive
- Limited throughput

### 2. Direct Queue Access (Production)
**Use Case**: High-volume internal services (Postal Service)

**Pros**:
- High throughput (10,000+ messages/second)
- Low latency (~10ms)
- Built-in reliability
- Cost effective

**Cons**:
- Requires Service Bus integration
- More complex setup

**Implementation**:
```python
# Postal service direct integration
class PostalServiceQueueClient:
    def __init__(self, connection_string):
        self.client = ServiceBusClient.from_connection_string(connection_string)
        self.sender = self.client.get_queue_sender(queue_name="zxing-input-queue")
    
    async def submit_barcode_job(self, image_data, metadata):
        message = ServiceBusMessage({
            "jobId": f"postal_{uuid.uuid4()}",
            "imageData": base64.b64encode(image_data).decode(),
            "metadata": metadata,
            "source": "postal_service",
            "priority": "high"
        })
        
        await self.sender.send_messages(message)
        return message.job_id
```

## Auto-Scaling Strategy

### Queue Depth Monitoring
```python
class AutoScaler:
    async def monitor_queue_depth(self):
        while True:
            queue_depth = await self.get_queue_depth()
            
            if queue_depth > 100:  # High load
                await self.scale_up_workers()
            elif queue_depth < 10:  # Low load
                await self.scale_down_workers()
            
            await asyncio.sleep(30)  # Check every 30 seconds
```

### Scaling Rules
- **Scale Up**: When queue depth > 100 messages
- **Scale Down**: When queue depth < 10 messages
- **Min Workers**: 2 (always available)
- **Max Workers**: 20 (cost control)

## Error Handling & Reliability

### Failure Detection
- **Message Delivery Count**: Automatic retry tracking
- **Lock Duration**: 5 minutes per worker
- **Max Retries**: 3 attempts per message
- **Dead Letter Queue**: Failed messages for investigation

### Error Classification
```python
class TemporaryError(Exception):
    """Errors that might succeed on retry"""
    pass

class PermanentError(Exception):
    """Errors that will never succeed"""
    pass

# Examples
class NetworkTimeoutError(TemporaryError):
    pass

class ImageCorruptedError(PermanentError):
    pass
```

### Monitoring & Alerting
- **Queue Metrics**: Message count, dead letter count
- **Worker Health**: Processing time, error rate
- **System Metrics**: CPU, memory, network usage
- **Business Metrics**: Throughput, success rate

## Cost Analysis

### Monthly Costs (1M packages/day)

| Component | Cost | Notes |
|-----------|------|-------|
| **ACI Containers (12)** | $360 | Auto-scaling |
| **App Service** | $50 | HTTP API |
| **Service Bus** | $20 | Message queuing |
| **Load Balancer** | $18 | Traffic distribution |
| **Blob Storage** | $276 | Image storage |
| **Cosmos DB** | $21 | Result storage |
| **Total** | **$745** | **$0.000745 per package** |

### Cost Optimization
- **Lifecycle Policies**: Move old images to cheaper tiers
- **Auto-scaling**: Pay only for active processing
- **Reserved Instances**: 20% discount for predictable workloads
- **Spot Instances**: 60% discount for batch processing

## Security Considerations

### Network Security
- **Private Endpoints**: Service Bus accessible only from authorized networks
- **VPN/ExpressRoute**: Secure connection for postal service
- **Firewall Rules**: IP whitelisting for production access

### Authentication & Authorization
- **Service Bus**: Connection strings with specific permissions
- **Azure AD**: Role-based access control
- **API Keys**: For external HTTP access

### Data Protection
- **Encryption**: At rest and in transit
- **Data Residency**: Compliance with regional requirements
- **Audit Logging**: Track all access and operations

## Deployment Strategy

### Phase 1: Proof of Concept
- Deploy 3 containers
- Test HTTP API
- Validate processing performance
- **Expected**: ~3.7 requests/second

### Phase 2: Production Ready
- Deploy 12 containers
- Implement auto-scaling
- Add monitoring and alerting
- **Expected**: ~14.8 requests/second

### Phase 3: Direct Integration
- Set up Service Bus direct access
- Integrate with postal service
- Optimize for high-volume processing
- **Expected**: 20+ requests/second

### Phase 4: Optimization
- Fine-tune auto-scaling rules
- Implement cost optimization
- Add advanced monitoring
- **Expected**: 25+ requests/second

## Monitoring & Observability

### Key Metrics
- **Throughput**: Requests per second
- **Latency**: Processing time per request
- **Error Rate**: Failed requests percentage
- **Queue Depth**: Messages waiting for processing
- **Worker Health**: CPU, memory, processing time

### Dashboards
- **Real-time Performance**: Current throughput and latency
- **Historical Trends**: Performance over time
- **Error Analysis**: Failed jobs and error types
- **Cost Tracking**: Resource usage and costs

### Alerting
- **High Error Rate**: >5% failure rate
- **Queue Backup**: >500 messages in queue
- **Worker Failure**: Worker health check failures
- **Cost Threshold**: Monthly cost exceeds budget

## Conclusion

This architecture provides a scalable, reliable, and cost-effective solution for processing 1 million packages per day. The combination of Azure Service Bus for message queuing, Azure Container Instances for processing, and Azure Storage for data persistence creates a robust system that can handle high-volume barcode processing with automatic scaling and comprehensive monitoring.

The dual integration approach (HTTP API + direct queue access) ensures flexibility for different client types while maintaining optimal performance for high-volume production workloads.
