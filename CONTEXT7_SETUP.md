# Context7 MCP Setup Guide

## Overview
This project now includes Context7 MCP server integration for up-to-date documentation access.

## Setup Instructions

### 1. Get Context7 API Key
1. Visit [context7.com/dashboard](https://context7.com/dashboard)
2. Create an account or sign in
3. Generate an API key

### 2. Configure API Key
Replace `YOUR_API_KEY_HERE` in `.cursor/mcp.json` with your actual API key:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "your_actual_api_key_here"
      }
    }
  }
}
```

### 3. Alternative Configuration Methods

#### Method 1: Environment Variable
Set the environment variable in your shell:
```bash
export CONTEXT7_API_KEY="your_api_key_here"
```

#### Method 2: Direct CLI Arguments
Modify `.cursor/mcp.json` to use CLI arguments:
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "your_api_key_here"]
    }
  }
}
```

## Available Tools

Context7 MCP provides these tools:

- `resolve-library-id`: Resolves a general library name into a Context7-compatible library ID
- `get-library-docs`: Fetches documentation for a library using a Context7-compatible library ID

## Usage Examples

### For zxing-cpp Documentation
```
use library /zxing-cpp/zxing-cpp for API and docs
```

### For Other Libraries
```
use library /mongodb/docs for MongoDB documentation
use library /vercel/next.js for Next.js documentation
```

## Troubleshooting

### Module Not Found Errors
If you encounter `ERR_MODULE_NOT_FOUND`, try using `bunx` instead of `npx`:

```json
{
  "mcpServers": {
    "context7": {
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```

### ESM Resolution Issues
For errors like `Error: Cannot find module 'uriTemplate.js'`, try:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-vm-modules", "@upstash/context7-mcp@1.0.6"]
    }
  }
}
```

### TLS/Certificate Issues
Use the `--experimental-fetch` flag:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "--node-options=--experimental-fetch", "@upstash/context7-mcp"]
    }
  }
}
```

## Testing the Integration

After setup, you can test the integration by asking questions like:
- "How do I use zxing-cpp for barcode detection?"
- "What are the latest zxing-cpp API methods?"
- "Show me zxing-cpp configuration options"

The Context7 MCP server will automatically fetch the latest documentation.
