// MCP连接器
const net = require('net');

class MCPConnector {
  constructor(config) {
    this.config = config;
    this.client = new net.Socket();
  }

  async connect() {
    return new Promise((resolve) => {
      this.client.connect(this.config.port, this.config.host, () => {
        console.log('已连接到MCP服务器');
        resolve(true);
      });

      this.client.on('error', (err) => {
        console.error('连接MCP服务器失败:', err);
        resolve(false);
      });
    });
  }

  async sendMessage(message) {
    return new Promise((resolve) => {
      this.client.write(JSON.stringify(message) + '\n', (err) => {
        if (err) {
          console.error('发送消息失败:', err);
          resolve({ status: 'error', message: err.message });
        } else {
          console.log('消息已发送');
          resolve({ status: 'success', message: '消息已发送' });
        }
      });
    });
  }

  async disconnect() {
    return new Promise((resolve) => {
      this.client.end(() => {
        console.log('已断开与MCP服务器的连接');
        resolve();
      });
    });
  }
}

const mcpConfig = {
  host: 'localhost',
  port: 3000
};

module.exports = { MCPConnector, mcpConfig };