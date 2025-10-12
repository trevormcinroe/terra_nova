const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT_DIR = __dirname;  // Serve files relative to project folder
const GAMESTATE_DIR = path.join(ROOT_DIR, './saved_games/');  // .json files live here
const PORT = 3001;

const server = http.createServer((req, res) => {
  if (req.url === '/list-saved-games') {
    fs.readdir(GAMESTATE_DIR, (err, files) => {
      if (err) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Failed to read gamestates directory.');
        return;
      }
      const jsonFiles = files.filter(file => file.endsWith('.ndjson.gz')).map(file => path.join('../saved_games/', file));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(jsonFiles));
    });
  } else {
    // Serve static files
    // Normalize the path to prevent directory traversal attacks
    let safePath = path.normalize(req.url).replace(/^(\.\.[\/\\])+/, '');
    let filePath = path.join(ROOT_DIR, safePath === '/' ? '/index_webgl.html' : safePath);

    const extname = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.html': 'text/html',
      '.js': 'text/javascript',
      '.json': 'application/json',
      '.css': 'text/css',
      '.png': 'image/png',
      '.jpg': 'image/jpg',
      '.gif': 'image/gif',
    };
    const contentType = mimeTypes[extname] || 'application/octet-stream';

    fs.readFile(filePath, (err, content) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
      } else {
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(content);
      }
    });
  }
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
