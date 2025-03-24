FROM node:22.14.0

WORKDIR /app

COPY package.json .

COPY package-lock.json .

RUN npm install --production

COPY . .

EXPOSE 9000

CMD ["node", "custom_server.js"]