#!/bin/bash

# Register users
echo "Registering user1..."
curl -X POST http://35.188.130.234:5000/register -d "username=testuser&password=testpassword"
echo -e "\n"

echo "Registering user2...should fail since the user already exists"
curl -X POST http://35.188.130.234:5000/register -d "username=testuser&password=testpassword"
echo -e "\n"

# Login users
echo "Logging in user1 with incorrect credentials..."
curl -X POST http://35.188.130.234:5000/login -d "username=user1&password=test"
echo -e "\n"

echo "Logging in user2 with correct credentials..."
curl -X POST http://35.188.130.234:5000/login -d "username=testuser1&password=testpassword"
echo -e "\n"

# Fetch weather
echo "Fetching weather for Accra..."
curl -X POST http://35.188.130.234:5000/fetch_weather -H "Content-Type: application/json" -d '{"location": "Accra"}'
echo -e "\n"

# Predict water availability
echo "Predicting water availability with data [1.0, 2.0, 3.0, 4, 4]..."
curl -X POST http://35.188.130.234:5000/predict_water_availability -H "Content-Type: application/json" -d '{"data": [1.0, 2.0, 3.0,4,4]}'
echo -e "\n"

echo "Predicting water availability with missing values for  data [1.0, 2.0, 3.4]..."
curl -X POST http://35.188.130.234:5000/predict_water_availability -H "Content-Type: application/json" -d '{"data": [1.0, 2.0, 3.4]}'
echo -e "\n"

# Chat with chatbot
echo "Chatting with the chatbot: 'What is climate?'"
curl -X POST http://35.188.130.234:5000/chat -H "Content-Type: application/json" -d '{"input_text": "What is climate?"}'
echo -e "\n"

echo "All tests completed."
