package main

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/streadway/amqp"
)

// RabbitMQClient implements QueueClient using RabbitMQ
type RabbitMQClient struct {
	conn        *amqp.Connection
	channel     *amqp.Channel
	inputQueue  string
	outputQueue string
	mu          sync.RWMutex
	closed      bool
	consumerTag string
	msgs        <-chan amqp.Delivery
}

// NewRabbitMQClient creates a new RabbitMQ client
func NewRabbitMQClient(url, inputQueue, outputQueue string) (*RabbitMQClient, error) {
	// Connect to RabbitMQ
	conn, err := amqp.Dial(url)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RabbitMQ: %v", err)
	}

	// Create channel
	channel, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to open channel: %v", err)
	}

	client := &RabbitMQClient{
		conn:        conn,
		channel:     channel,
		inputQueue:  inputQueue,
		outputQueue: outputQueue,
		consumerTag: fmt.Sprintf("consumer-%d", time.Now().UnixNano()),
	}

	// Declare queues
	if err := client.declareQueues(); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to declare queues: %v", err)
	}

	// Create consumer once
	if err := client.createConsumer(); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to create consumer: %v", err)
	}

	return client, nil
}

// declareQueues declares the input and output queues
func (r *RabbitMQClient) declareQueues() error {
	// Declare input queue (durable, persistent)
	_, err := r.channel.QueueDeclare(
		r.inputQueue, // name
		true,         // durable
		false,        // delete when unused
		false,        // exclusive
		false,        // no-wait
		nil,          // arguments
	)
	if err != nil {
		return fmt.Errorf("failed to declare input queue: %v", err)
	}

	// Declare output queue (durable, persistent)
	_, err = r.channel.QueueDeclare(
		r.outputQueue, // name
		true,          // durable
		false,         // delete when unused
		false,         // exclusive
		false,         // no-wait
		nil,           // arguments
	)
	if err != nil {
		return fmt.Errorf("failed to declare output queue: %v", err)
	}

	return nil
}

// createConsumer creates a consumer for the input queue
func (r *RabbitMQClient) createConsumer() error {
	// Set QoS to process one message at a time
	err := r.channel.Qos(1, 0, false)
	if err != nil {
		return fmt.Errorf("failed to set QoS: %v", err)
	}

	// Create consumer once
	msgs, err := r.channel.Consume(
		r.inputQueue,  // queue
		r.consumerTag, // consumer tag (unique identifier)
		false,         // auto-ack (manual ack)
		false,         // exclusive
		false,         // no-local
		false,         // no-wait
		nil,           // args
	)
	if err != nil {
		return fmt.Errorf("failed to register consumer: %v", err)
	}

	r.msgs = msgs
	return nil
}

// Receive gets a message from the input queue
func (r *RabbitMQClient) Receive() (*QueueMessage, error) {
	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return nil, ErrQueueClosed
	}
	r.mu.RUnlock()

	// Wait for message with timeout using existing consumer
	select {
	case msg := <-r.msgs:
		// Parse message body
		var queueMsg QueueMessage
		if err := json.Unmarshal(msg.Body, &queueMsg); err != nil {
			msg.Nack(false, false) // Reject message
			return nil, fmt.Errorf("failed to unmarshal message: %v", err)
		}

		// Acknowledge message
		msg.Ack(false)

		return &queueMsg, nil

	case <-time.After(100 * time.Millisecond):
		return nil, ErrNoMessage
	}
}

// SendResult sends a result to the output queue
func (r *RabbitMQClient) SendResult(result *ProcessingResult) error {
	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return ErrQueueClosed
	}
	r.mu.RUnlock()

	// Marshal result
	body, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %v", err)
	}

	// Publish message
	err = r.channel.Publish(
		"",            // exchange
		r.outputQueue, // routing key
		false,         // mandatory
		false,         // immediate
		amqp.Publishing{
			ContentType:   "application/json",
			Body:          body,
			DeliveryMode:  amqp.Persistent, // Make message persistent
			CorrelationId: result.JobID,    // Set correlation ID
		},
	)
	if err != nil {
		return fmt.Errorf("failed to publish result: %v", err)
	}

	return nil
}

// SendMessage sends a message to the input queue (for testing)
func (r *RabbitMQClient) SendMessage(msg QueueMessage) error {
	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return ErrQueueClosed
	}
	r.mu.RUnlock()

	// Marshal message
	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	// Publish message
	err = r.channel.Publish(
		"",           // exchange
		r.inputQueue, // routing key
		false,        // mandatory
		false,        // immediate
		amqp.Publishing{
			ContentType:   "application/json",
			Body:          body,
			DeliveryMode:  amqp.Persistent, // Make message persistent
			CorrelationId: msg.JobID,       // Set correlation ID
		},
	)
	if err != nil {
		return fmt.Errorf("failed to publish message: %v", err)
	}

	return nil
}

// GetResult gets a result from the output queue (for testing)
func (r *RabbitMQClient) GetResult() (*ProcessingResult, error) {
	r.mu.RLock()
	if r.closed {
		r.mu.RUnlock()
		return nil, ErrQueueClosed
	}
	r.mu.RUnlock()

	// Consume message
	msgs, err := r.channel.Consume(
		r.outputQueue, // queue
		"",            // consumer
		false,         // auto-ack (manual ack)
		false,         // exclusive
		false,         // no-local
		false,         // no-wait
		nil,           // args
	)
	if err != nil {
		return nil, fmt.Errorf("failed to register consumer: %v", err)
	}

	// Wait for message with timeout
	select {
	case msg := <-msgs:
		// Parse message body
		var result ProcessingResult
		if err := json.Unmarshal(msg.Body, &result); err != nil {
			msg.Nack(false, false) // Reject message
			return nil, fmt.Errorf("failed to unmarshal result: %v", err)
		}

		// Acknowledge message
		msg.Ack(false)

		return &result, nil

	case <-time.After(100 * time.Millisecond):
		return nil, ErrNoMessage
	}
}

// Close closes the RabbitMQ connection
func (r *RabbitMQClient) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if !r.closed {
		r.closed = true
		// Cancel consumer
		if r.consumerTag != "" {
			r.channel.Cancel(r.consumerTag, false)
		}
		r.channel.Close()
		r.conn.Close()
	}
	return nil
}

// Queue errors
var (
	ErrQueueClosed = fmt.Errorf("queue is closed")
	ErrNoMessage   = fmt.Errorf("no message available")
	ErrQueueFull   = fmt.Errorf("queue is full")
)
