# frontend/Dockerfile  (build context = project root via docker-compose)
# Stage 1: build React
FROM node:20-alpine AS builder
WORKDIR /build
COPY src/frontend/package*.json ./
RUN npm install
COPY src/frontend/ .
RUN npm run build

# Stage 2: serve con nginx
FROM nginx:alpine
COPY --from=builder /build/dist /usr/share/nginx/html
COPY deployment/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
