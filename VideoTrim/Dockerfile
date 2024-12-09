# Use official Node.js runtime as the base image
FROM node:18

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json separately to leverage Docker's layer caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Install Expo CLI and @expo/ngrok globally for Expo development and debugging
RUN npm install -g expo-cli @expo/ngrok

# Copy the application files
COPY . .

# Disable telemetry to avoid unnecessary logs and analytics during builds
ENV EXPO_NO_TELEMETRY=1
ENV CI=true

# Expose Expo and Metro bundler ports
EXPOSE 19000 8081

# Start the Expo development server
CMD ["npx", "expo", "start", "--tunnel"]
