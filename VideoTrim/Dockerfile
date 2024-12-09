# Use an official Node.js runtime as a parent image
FROM node:18-slim

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json first to leverage Docker caching
COPY package*.json /app/

# Install dependencies
RUN npm install

# Copy the rest of the application files into the container
COPY . /app

# Install Expo CLI globally
RUN npm install -g expo-cli

# Ensure that the `expo-cli` works non-interactively
ENV CI=true

# Make ports 19000, 19001, and 8081 available to the world outside this container
EXPOSE 19000 19001 8081

# Start the Expo CLI
CMD ["npx", "expo", "start", "--tunnel"]
