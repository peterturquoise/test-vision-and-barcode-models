package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"

	"github.com/streadway/amqp"
)

// Simple Test Client
type SimpleTestClient struct {
	conn      *amqp.Connection
	channel   *amqp.Channel
	imageData []byte
}

func NewSimpleTestClient(rabbitmqURL, imagePath string) (*SimpleTestClient, error) {
	conn, err := amqp.Dial(rabbitmqURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RabbitMQ: %v", err)
	}

	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to open channel: %v", err)
	}

	// Declare queues
	_, err = channel.QueueDeclare("image-processing", true, false, false, false, nil)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare input queue: %v", err)
	}

	_, err = channel.QueueDeclare("results", true, false, false, false, nil)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare output queue: %v", err)
	}

	// Load test image
	imageData, err := ioutil.ReadFile(imagePath)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to load test image: %v", err)
	}

	return &SimpleTestClient{
		conn:      conn,
		channel:   channel,
		imageData: imageData,
	}, nil
}

func (c *SimpleTestClient) RunTest(duration time.Duration) {
	log.Printf("🚀 Starting Simple RabbitMQ test for %v", duration)

	startTime := time.Now()
	endTime := startTime.Add(duration)
	jobCounter := 0

	// Start result collector
	go c.collectResults()

	// Submit jobs continuously
	for time.Now().Before(endTime) {
		message := ImageMessage{
			JobID:     fmt.Sprintf("job_%d_%d", time.Now().UnixNano(), jobCounter),
			ImageData: c.imageData,
			Pattern:   "",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		}

		body, _ := json.Marshal(message)
		err := c.channel.Publish("", "image-processing", false, false, amqp.Publishing{
			ContentType:   "application/json",
			Body:          body,
			DeliveryMode:  amqp.Persistent,
			CorrelationId: message.JobID,
		})

		if err != nil {
			log.Printf("Failed to send message: %v", err)
			time.Sleep(100 * time.Millisecond)
			continue
		}

		jobCounter++
		time.Sleep(10 * time.Millisecond)
	}

	log.Printf("📤 Submitted %d jobs total", jobCounter)

	// Wait for remaining results
	time.Sleep(2 * time.Second)
}

func (c *SimpleTestClient) collectResults() {
	// Consume results
	msgs, err := c.channel.Consume("results", "", true, false, false, false, nil)
	if err != nil {
		log.Printf("Failed to register result consumer: %v", err)
		return
	}

	var results []ResultMessage

	for msg := range msgs {
		var result ResultMessage
		if err := json.Unmarshal(msg.Body, &result); err != nil {
			log.Printf("Failed to unmarshal result: %v", err)
			continue
		}

		results = append(results, result)

		log.Printf("📊 Result: Job %s - %s (%.3fs) - %d barcodes",
			result.JobID, result.Status, result.ProcessingTime, len(result.Barcodes))
	}

	// Print statistics
	c.printStatistics(results)
}

func (c *SimpleTestClient) printStatistics(results []ResultMessage) {
	if len(results) == 0 {
		log.Println("No results received")
		return
	}

	successful := 0
	failed := 0
	totalBarcodes := 0
	var totalProcessingTime float64

	for _, result := range results {
		if result.Status == "success" {
			successful++
			totalBarcodes += len(result.Barcodes)
		} else {
			failed++
		}
		totalProcessingTime += result.ProcessingTime
	}

	avgProcessingTime := totalProcessingTime / float64(len(results))

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📊 SIMPLE RABBITMQ TEST RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("📈 Total Jobs: %d\n", len(results))
	fmt.Printf("✅ Successful: %d\n", successful)
	fmt.Printf("❌ Failed: %d\n", failed)
	fmt.Printf("📊 Success Rate: %.1f%%\n", float64(successful)/float64(len(results))*100)
	fmt.Printf("🔍 Total Barcodes Found: %d\n", totalBarcodes)
	fmt.Printf("⏱️  Average Processing Time: %.3fs\n", avgProcessingTime)
	fmt.Println(strings.Repeat("=", 60))
}

func (c *SimpleTestClient) Close() {
	c.channel.Close()
	c.conn.Close()
}

func main() {
	if len(os.Args) < 2 {
		log.Println("Usage: go run simple_client.go <image_path> [duration]")
		log.Println("Example: go run simple_client.go test_image.jpg 30s")
		return
	}

	imagePath := os.Args[1]
	duration := 20 * time.Second

	if len(os.Args) > 2 {
		if parsedDuration, err := time.ParseDuration(os.Args[2]); err == nil {
			duration = parsedDuration
		} else {
			log.Printf("Invalid duration '%s', using default 20s", os.Args[2])
		}
	}

	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://admin:admin123@localhost:5672/"
	}

	log.Printf("RabbitMQ URL: %s", rabbitmqURL)

	client, err := NewSimpleTestClient(rabbitmqURL, imagePath)
	if err != nil {
		log.Fatalf("Failed to create test client: %v", err)
	}
	defer client.Close()

	client.RunTest(duration)
}
