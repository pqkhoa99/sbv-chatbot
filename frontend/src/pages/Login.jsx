import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    if (isRegisterMode) {
      const result = await register(username, password, name);
      
      if (result.success) {
        setSuccess('Đăng ký thành công! Vui lòng đăng nhập.');
        setUsername('');
        setPassword('');
        setName('');
        setIsRegisterMode(false);
      } else {
        setError(result.error);
      }
    } else {
      const result = await login(username, password);
      
      if (result.success) {
        navigate('/chat');
      } else {
        setError(result.error);
      }
    }
    
    setLoading(false);
  };
  
  const toggleMode = () => {
    setIsRegisterMode(!isRegisterMode);
    setError('');
    setSuccess('');
    setUsername('');
    setPassword('');
    setName('');
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-logo">
          <img src="/images/sbv-logo.jpg" alt="Logo NHNN" />
        </div>
        
        <div className="login-header">
          <h1>Chatbot Hỏi Đáp Pháp Luật</h1>
          <h2>Ngân Hàng Nhà Nước Việt Nam</h2>
          <p className="login-mode-title">{isRegisterMode ? 'Đăng ký tài khoản' : 'Đăng nhập hệ thống'}</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          {isRegisterMode && (
            <div className="form-group">
              <label htmlFor="name">Họ và tên</label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Nhập họ và tên"
                required
                disabled={loading}
              />
            </div>
          )}
          
          <div className="form-group">
            <label htmlFor="username">Tên đăng nhập</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Nhập tên đăng nhập"
              required
              disabled={loading}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Mật khẩu</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Nhập mật khẩu"
              required
              disabled={loading}
            />
          </div>
          
          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}
          
          <button type="submit" className="login-button" disabled={loading}>
            {loading ? (isRegisterMode ? 'Đang đăng ký...' : 'Đang đăng nhập...') : (isRegisterMode ? 'Đăng ký' : 'Đăng nhập')}
          </button>
        </form>
        
        <div className="login-footer">
          <p className="register-text">
            {isRegisterMode ? (
              <>Đã có tài khoản? <span className="register-link" onClick={toggleMode}>Đăng nhập</span> ngay</>
            ) : (
              <>Bạn chưa có tài khoản? <span className="register-link" onClick={toggleMode}>Đăng ký</span> ngay</>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
