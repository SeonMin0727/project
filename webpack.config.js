const path = require('path');

module.exports = {
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '@lib': path.resolve(__dirname, 'src/lib'),
      // 기타 필요한 경로 매핑
    },
  },
  // 기타 webpack 설정
};