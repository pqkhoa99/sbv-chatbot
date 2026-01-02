import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  login: async (username, password) => {
    const response = await api.post('/api/auth/login', { username, password });
    return response.data;
  },
  
  register: async (username, password, name) => {
    const response = await api.post('/api/auth/register', { username, password, name });
    return response.data;
  },
};

export const chatAPI = {
  sendMessage: async (conversation_id, question) => {
    const response = await api.post('/api/chat', { 
      conversation_id,
      question
    });
    return response.data;
  },
  
  getModels: async () => {
    const response = await api.get('/api/models');
    return response.data;
  },
  
  createConversation: async (title, model) => {
    const response = await api.post('/api/conversations', { title, model });
    return response.data;
  },
  
  getConversations: async () => {
    const response = await api.get('/api/conversations');
    return response.data;
  },
  
  getConversation: async (id) => {
    const response = await api.get(`/api/conversations/${id}`);
    return response.data;
  },
  
  deleteConversation: async (id) => {
    const response = await api.delete(`/api/conversations/${id}`);
    return response.data;
  },
  
  updateConversationTitle: async (id, title) => {
    const response = await api.patch(`/api/conversations/${id}/title`, { title });
    return response.data;
  },
};

export default api;
