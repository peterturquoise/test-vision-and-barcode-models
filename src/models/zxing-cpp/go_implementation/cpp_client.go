package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"github.com/streadway/amqp"
)

func main() {
	fmt.Println("🚀 C++ ZXing Client - Sending image to RabbitMQ")
	
	// Connect to RabbitMQ
	conn, err := amqp.Dial("amqp://guest:guest@localhost:5672/")
	if err != nil {
		log.Fatalf("Failed to connect to RabbitMQ: %v", err)
	}
	defer conn.Close()

	ch, err := conn.Channel()
	if err != nil {
		log.Fatalf("Failed to open channel: %v", err)
	}
	defer ch.Close()

	// Declare queues
	imageQ, err := ch.QueueDeclare(
		"image_queue", // name
		false,         // durable
		false,         // delete when unused
		false,         // exclusive
		false,         // no-wait
		nil,           // arguments
	)
	if err != nil {
		log.Fatalf("Failed to declare image queue: %v", err)
	}

	resultQ, err := ch.QueueDeclare(
		"result_queue", // name
		false,          // durable
		false,          // delete when unused
		false,          // exclusive
		false,          // no-wait
		nil,            // arguments
	)
	if err != nil {
		log.Fatalf("Failed to declare result queue: %v", err)
	}

	// Read test image
	imageData, err := ioutil.ReadFile("test_image.jpeg")
	if err != nil {
		log.Fatalf("Failed to read test image: %v", err)
	}

	fmt.Printf("📸 Sending image (%d bytes) to image_queue\n", len(imageData))

	// Send image to queue
	err = ch.Publish(
		"",           // exchange
		imageQ.Name, // routing key
		false,       // mandatory
		false,       // immediate
		amqp.Publishing{
			ContentType: "image/jpeg",
			Body:        imageData,
		},
	)
	if err != nil {
		log.Fatalf("Failed to publish image: %v", err)
	}

	fmt.Println("✅ Image sent! Waiting for results...")

	// Consume results
	msgs, err := ch.Consume(
		resultQ.Name, // queue
		"",           // consumer
		true,         // auto-ack
		false,        // exclusive
		false,        // no-local
		false,        // no-wait
		nil,          // args
	)
	if err != nil {
		log.Fatalf("Failed to register consumer: %v", err)
	}

	// Wait for result with timeout
	select {
	case d := <-msgs:
		fmt.Printf("📥 Received result: %s\n", string(d.Body))
	case <-time.After(30 * time.Second):
		fmt.Println("⏰ Timeout waiting for result")
	}

	fmt.Println("✅ Client complete!")
}
