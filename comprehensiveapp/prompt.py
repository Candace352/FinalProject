# Flask endpoint for rainfall recommendations
@app.route('/rainfall-recommendations', methods=['POST'])
def rainfall_recommendations():
    # Get the average rainfall and area from the request
    data = request.get_json()
    if not data or 'average_rainfall' not in data or 'area' not in data:
        return jsonify({"error": "Please provide 'average_rainfall' and 'area' in the request body"}), 400

    try:
        average_rainfall = float(data['average_rainfall'])
        area = data['area']
    except ValueError:
        return jsonify({"error": "'average_rainfall' must be a valid number and 'area' must be a string"}), 400

    # Define the message prompt for the model
    messages = [
        {
            "role": "user",
            "content": (
                f"The average rainfall in the {area} region is {average_rainfall} millimeters. "
                f"Explain what this average rainfall means in relation to the {area} region, considering its geography, climate, and typical water needs. "
                f"Based on this, provide recommendations for water preservation, including specific dos and don'ts to help the community manage water resources effectively."
            )
        }
    ]

    # Query the Hugging Face model
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=500
        )
        recommendations = completion.choices[0].message["content"]
    except Exception as e:
        return jsonify({"error": f"An error occurred while generating recommendations: {str(e)}"}), 500

    # Return the recommendations
    return jsonify({
        "average_rainfall": average_rainfall,
        "area": area,
        "recommendations": recommendations
    })
