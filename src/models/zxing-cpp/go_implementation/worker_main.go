package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Get number of workers from environment or default to 4
	workers := 4
	if envWorkers := os.Getenv("WORKERS"); envWorkers != "" {
		if n, err := fmt.Sscanf(envWorkers, "%d", &workers); err != nil || n != 1 {
			log.Printf("Invalid WORKERS value, using default: %d", workers)
		}
	}

	log.Printf("Starting ZXing Queue Worker with %d workers", workers)

	// Create in-memory queues for testing (in production, these would be Azure Service Bus)
	inputQueue := NewMemoryQueue(1000)
	outputQueue := NewMemoryQueue(1000)

	// Create and start workers
	workerList := make([]*QueueWorker, workers)
	for i := 0; i < workers; i++ {
		worker := NewQueueWorker(i, inputQueue, outputQueue)
		workerList = append(workerList, worker)
		go worker.Start()
	}

	// Set up graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for shutdown signal
	<-sigChan
	log.Println("Received shutdown signal, stopping workers...")

	// Stop all workers
	for _, worker := range workerList {
		if worker != nil {
			worker.Stop()
		}
	}

	// Close queues
	inputQueue.Close()
	outputQueue.Close()

	log.Println("All workers stopped, exiting")
}
