import requests

# API endpoint
URL = "https://waterservices.usgs.gov/nwis/iv/"

# Parameters for the API
params = {
    'format': 'json',
    'sites': '412611081210600',  # Replace with the site ID for your region
    'parameterCd': '72019',  # Code for groundwater level
    'period': 'P1D',  # Past 1 day of data
}

response = requests.get(URL, params=params)
print(response.json())
# if response.status_code == 200:
#     data = response.json()
#     groundwater_level = data['value']['timeSeries'][0]['values'][0]['value']
#     print(f"Groundwater Level: {groundwater_level}")
# else:
#     print("Error fetching data:", response.status_code)
