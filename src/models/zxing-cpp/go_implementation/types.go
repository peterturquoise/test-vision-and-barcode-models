package main

// Message types (shared between worker and client)
type ImageMessage struct {
	JobID     string `json:"jobId"`
	ImageData []byte `json:"imageData"`
	Pattern   string `json:"pattern"`
	Timestamp string `json:"timestamp"`
}

type ResultMessage struct {
	JobID          string          `json:"jobId"`
	Status         string          `json:"status"`
	Barcodes       []BarcodeResult `json:"barcodes"`
	ProcessingTime float64         `json:"processingTime"`
	Error          string          `json:"error,omitempty"`
	Timestamp      string          `json:"timestamp"`
}

type BarcodeResult struct {
	Value       string  `json:"value"`
	BarcodeType string  `json:"barcode_type"`
	Bbox        []int   `json:"bbox"`
	Rotation    int     `json:"rotation"`
	HasError    bool    `json:"has_error"`
	ErrorType   *string `json:"error_type"`
	Confidence  float64 `json:"confidence"`
	Approach    string  `json:"approach"`
}
