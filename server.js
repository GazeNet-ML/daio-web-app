const express = require('express');
const next = require('next');
const multer = require('multer');
const path = require('path');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

const upload = multer({ dest: 'public/uploads/' });

app.prepare().then(() => {
  const server = express();

  server.post('/api/upload', upload.single('video'), (req, res) => {
    res.status(200).json({ filePath: `/uploads/${req.file.filename}` });
  });

  server.all('*', (req, res) => {
    return handle(req, res);
  });

  server.listen(3000, (err) => {
    if (err) throw err;
    console.log('> Ready on http://localhost:3000');
  });
});
