package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Get configuration from environment
	workers := 4
	if envWorkers := os.Getenv("WORKERS"); envWorkers != "" {
		if n, err := fmt.Sscanf(envWorkers, "%d", &workers); err != nil || n != 1 {
			log.Printf("Invalid WORKERS value, using default: %d", workers)
		}
	}

	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://admin:admin123@localhost:5672/"
	}

	inputQueue := "image-processing"
	outputQueue := "results"

	log.Printf("Starting ZXing RabbitMQ Worker with %d workers", workers)
	log.Printf("RabbitMQ URL: %s", rabbitmqURL)
	log.Printf("Input Queue: %s", inputQueue)
	log.Printf("Output Queue: %s", outputQueue)

	// Create RabbitMQ client
	queueClient, err := NewRabbitMQClient(rabbitmqURL, inputQueue, outputQueue)
	if err != nil {
		log.Fatalf("Failed to create RabbitMQ client: %v", err)
	}
	defer queueClient.Close()

	// Create and start workers
	workerList := make([]*QueueWorker, workers)
	for i := 0; i < workers; i++ {
		worker := NewQueueWorker(i, queueClient, queueClient)
		workerList = append(workerList, worker)
		go worker.Start()
	}

	// Set up graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Println("Workers started. Press Ctrl+C to stop...")

	// Wait for shutdown signal
	<-sigChan
	log.Println("Received shutdown signal, stopping workers...")

	// Stop all workers
	for _, worker := range workerList {
		if worker != nil {
			worker.Stop()
		}
	}

	log.Println("All workers stopped, exiting")
}
