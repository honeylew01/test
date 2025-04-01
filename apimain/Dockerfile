
# Use python image
FROM python:3.10

# Set working directory
WORKDIR /app


COPY . /app


RUN pip install uvicorn pandas scikit-learn fastapi pydantic


# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "kmeans_api:app", "--host", "0.0.0.0", "--port", "8000"]