# =====================================================================
# Stage 1: "builder" - Build the Next.js application
# This stage has all the dev dependencies and source code.
# =====================================================================
FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependency definitions
COPY package.json package-lock.json ./

# Install all dependencies (including dev dependencies needed for the build)
RUN npm install

# Copy the rest of the source code
COPY . .

# Build the production-ready application
RUN npm run build


# =====================================================================
# Stage 2: "runner" - The final, lightweight production image
# This stage starts fresh and copies ONLY what's needed to run the app.
# =====================================================================
FROM node:18-alpine AS runner

WORKDIR /app

# Create a non-root user for better security
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Copy only the necessary files from the "builder" stage
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Switch to the non-root user
USER nextjs

EXPOSE 3000

# The command to start the production server
CMD ["npm", "start"]