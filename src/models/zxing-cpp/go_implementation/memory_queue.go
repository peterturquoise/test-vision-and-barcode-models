package main

import (
	"fmt"
	"sync"
	"time"
)

// MemoryQueue implements QueueClient using in-memory channels
type MemoryQueue struct {
	messages chan QueueMessage
	results  chan ProcessingResult
	mu       sync.RWMutex
	closed   bool
}

// NewMemoryQueue creates a new in-memory queue
func NewMemoryQueue(bufferSize int) *MemoryQueue {
	return &MemoryQueue{
		messages: make(chan QueueMessage, bufferSize),
		results:  make(chan ProcessingResult, bufferSize),
	}
}

// Receive gets a message from the input queue
func (q *MemoryQueue) Receive() (*QueueMessage, error) {
	q.mu.RLock()
	if q.closed {
		q.mu.RUnlock()
		return nil, ErrQueueClosed
	}
	q.mu.RUnlock()

	select {
	case msg := <-q.messages:
		return &msg, nil
	case <-time.After(100 * time.Millisecond):
		return nil, ErrNoMessage
	}
}

// SendResult sends a result to the output queue
func (q *MemoryQueue) SendResult(result *ProcessingResult) error {
	q.mu.RLock()
	if q.closed {
		q.mu.RUnlock()
		return ErrQueueClosed
	}
	q.mu.RUnlock()

	select {
	case q.results <- *result:
		return nil
	case <-time.After(1 * time.Second):
		return ErrQueueFull
	}
}

// SendMessage sends a message to the input queue (for testing)
func (q *MemoryQueue) SendMessage(msg QueueMessage) error {
	q.mu.RLock()
	if q.closed {
		q.mu.RUnlock()
		return ErrQueueClosed
	}
	q.mu.RUnlock()

	select {
	case q.messages <- msg:
		return nil
	case <-time.After(1 * time.Second):
		return ErrQueueFull
	}
}

// GetResult gets a result from the output queue (for testing)
func (q *MemoryQueue) GetResult() (*ProcessingResult, error) {
	q.mu.RLock()
	if q.closed {
		q.mu.RUnlock()
		return nil, ErrQueueClosed
	}
	q.mu.RUnlock()

	select {
	case result := <-q.results:
		return &result, nil
	case <-time.After(100 * time.Millisecond):
		return nil, ErrNoMessage
	}
}

// Close closes the queue
func (q *MemoryQueue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if !q.closed {
		q.closed = true
		close(q.messages)
		close(q.results)
	}
	return nil
}

// Queue errors
var (
	ErrQueueClosed = fmt.Errorf("queue is closed")
	ErrNoMessage   = fmt.Errorf("no message available")
	ErrQueueFull   = fmt.Errorf("queue is full")
)
