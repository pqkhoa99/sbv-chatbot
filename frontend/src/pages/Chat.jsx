import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { chatAPI } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import './Chat.css';

const Chat = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedModel, setSelectedModel] = useState('sbv-lawgraph');
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [deleteConfirmation, setDeleteConfirmation] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }
    
    loadModels();
    loadConversations();
  }, [user, navigate]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [inputValue]);

  const loadModels = async () => {
    try {
      const data = await chatAPI.getModels();
      setModels(data);
    } catch (err) {
      console.error('Failed to load models:', err);
      setModels([]); // Set empty array on error to prevent undefined
    }
  };

  const loadConversations = async () => {
    try {
      const data = await chatAPI.getConversations();
      setConversations(data);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  };

  const loadConversation = async (conversationId) => {
    try {
      const data = await chatAPI.getConversation(conversationId);
      setCurrentConversation(data);
      setMessages(data.messages);
      setSelectedModel(data.model);
    } catch (err) {
      console.error('Failed to load conversation:', err);
    }
  };

  const createNewConversation = async (model = null) => {
    const modelToUse = model || selectedModel;
    const modelName = models?.find(m => m.id === modelToUse)?.name || modelToUse;
    
    try {
      const newConv = await chatAPI.createConversation(
        `Cuộc trò chuyện mới - ${modelName}`,
        modelToUse
      );
      setCurrentConversation(newConv);
      setMessages([]);
      setSelectedModel(modelToUse);
      setConversations(prev => [newConv, ...prev]);
      setShowModelSelector(false);
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  };

  const handleModelChange = (newModel) => {
    if (newModel !== selectedModel) {
      // Create new conversation when switching models
      createNewConversation(newModel);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;

    const userMessageContent = inputValue.trim();
    setInputValue('');
    setLoading(true);

    try {
      // Create new conversation if none exists
      let conversationToUse = currentConversation;
      if (!conversationToUse) {
        const modelToUse = selectedModel;
        const modelName = models?.find(m => m.id === modelToUse)?.name || modelToUse;
        
        const newConv = await chatAPI.createConversation(
          `Cuộc trò chuyện mới - ${modelName}`,
          modelToUse
        );
        conversationToUse = newConv;
        setCurrentConversation(newConv);
        setConversations(prev => [newConv, ...prev]);
        setShowModelSelector(false);
      }

      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: userMessageContent,
        created_at: new Date().toISOString(),
      };

      setMessages(prev => [...prev, userMessage]);

      const response = await chatAPI.sendMessage(conversationToUse.id, userMessageContent);
      
      setMessages(prev => [...prev, response.message]);
      
      // Update the current conversation's updated_at timestamp
      setConversations(prev => prev.map(conv => 
        conv.id === conversationToUse.id 
          ? { ...conv, updated_at: response.conversation_updated_at }
          : conv
      ));
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now(),
        role: 'assistant',
        content: 'Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau.',
        error: true,
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteConversation = async (convId, e) => {
    e.stopPropagation();
    setDeleteConfirmation(convId);
  };
  
  const confirmDelete = async () => {
    const convId = deleteConfirmation;
    setDeleteConfirmation(null);
    
    try {
      await chatAPI.deleteConversation(convId);
      setConversations(prev => prev.filter(c => c.id !== convId));
      if (currentConversation?.id === convId) {
        setCurrentConversation(null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const getModelBadge = (modelId) => {
    const model = models?.find(m => m.id === modelId);
    return model?.badge;
  };

  const getModelInfo = (modelId) => {
    return models?.find(m => m.id === modelId);
  };

  const CodeBlock = ({ node, inline, className, children, ...props }) => {
    const match = /language-(\w+)/.exec(className || '');
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
      navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    return !inline && match ? (
      <div className="code-block-wrapper">
        <div className="code-block-header">
          <span className="code-language">{match[1]}</span>
          <button className="copy-button" onClick={handleCopy}>
            {copied ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M20 6L9 17L4 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Đã sao chép
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M5 15H4C2.89543 15 2 14.1046 2 13V4C2 2.89543 2.89543 2 4 2H13C14.1046 2 15 2.89543 15 4V5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Sao chép
              </>
            )}
          </button>
        </div>
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      </div>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    );
  };

  return (
    <div className="chat-container-v2">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            className="sidebar-v2"
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          >
            <div className="sidebar-header-v2">
              <button className="new-chat-btn-v2" onClick={() => setShowModelSelector(!showModelSelector)}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                <span>Cuộc trò chuyện mới</span>
              </button>
              
              <AnimatePresence>
                {showModelSelector && (
                  <motion.div
                    className="model-selector-dropdown"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <div className="model-selector-header">Chọn mô hình:</div>
                    {models.map(model => (
                      <button
                        key={model.id}
                        className={`model-option ${model.id === 'sbv-lawgraph' ? 'model-option-highlight' : ''}`}
                        onClick={() => handleModelChange(model.id)}
                      >
                        <div className="model-option-content">
                          <div className="model-option-name">
                            {model.id === 'sbv-lawgraph' && <span className="top-badge">🏆</span>}
                            {model.name}
                            {model.badge && <span className="model-badge">{model.badge}</span>}
                          </div>
                          <div className="model-option-desc">{model.description}</div>
                        </div>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            
            <div className="conversations-list">
              {conversations.length === 0 ? (
                <div className="empty-conversations">
                  Chưa có cuộc trò chuyện nào
                </div>
              ) : (
                conversations.map(conv => (
                  <motion.div
                    key={conv.id}
                    className={`conversation-item ${currentConversation?.id === conv.id ? 'active' : ''}`}
                    onClick={() => loadConversation(conv.id)}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    whileHover={{ scale: 1.02, backgroundColor: 'rgba(0,0,0,0.02)' }}
                  >
                    <div className="conversation-content">
                      <div className="conversation-title">{conv.title}</div>
                      <div className="conversation-meta">
                        <span className="conversation-model">
                          <strong>{getModelInfo(conv.model)?.name}</strong>
                          {getModelBadge(conv.model) && (
                            <span className="model-badge-small">{getModelBadge(conv.model)}</span>
                          )}
                        </span>
                      </div>
                    </div>
                    <button
                      className="delete-conv-btn"
                      onClick={(e) => handleDeleteConversation(conv.id, e)}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </button>
                  </motion.div>
                ))
              )}
            </div>
            
            <div className="sidebar-footer-v2">
              <div className="user-info-v2">
                <div className="user-avatar-v2">
                  {user?.name?.charAt(0).toUpperCase() || user?.username?.charAt(0).toUpperCase()}
                </div>
                <div className="user-details-v2">
                  <div className="username-v2">{user?.name || user?.username}</div>
                  <div className="user-role-v2">{user?.role}</div>
                </div>
              </div>
              <button className="logout-btn-v2" onClick={handleLogout}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <path d="M9 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M16 17L21 12L16 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M21 12H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Đăng xuất
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="chat-main-v2">
        <div className="chat-header-v2">
          <button 
            className="sidebar-toggle-v2"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M3 12H21M3 6H21M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <div className="chat-header-content">
            <img src="/images/sbv-logo.jpg" alt="NHNN Logo" className="header-logo" />
            <h1>Chatbot Pháp Luật Ngân Hàng Nhà Nước Việt Nam</h1>
            {currentConversation && (
              <div className="current-model-badge">
                {getModelInfo(currentConversation.model)?.name}
                {getModelBadge(currentConversation.model) && (
                  <span className="badge-pill">{getModelBadge(currentConversation.model)}</span>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="messages-container-v2">
          {messages.length === 0 ? (
            <motion.div 
              className="empty-state-v2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="empty-icon">💬</div>
              {/* <img src="/images/sbv-logo.jpg" alt="NHNN Logo" className="welcome-logo" /> */}
              <h2>Chào mừng đến với Chatbot Pháp Luật Ngân Hàng Nhà Nước Việt Nam</h2>
              <p>Hãy bắt đầu cuộc trò chuyện bằng cách chọn mô hình và đặt câu hỏi</p>
              <div className="example-questions-v2">
                <div className="example-title">Ví dụ câu hỏi:</div>
                {[
                  'Có bao nhiêu hình thức mở tài khoản tại ngân hàng?',
                  'Những nhu cầu vay vốn nào sẽ không được tổ chức tín dụng xét duyệt hồ sơ cho vay?',
                  'Quy định về tỷ lệ an toàn vốn tối thiểu là bao nhiêu?'
                ].map((question, idx) => (
                  <motion.button
                    key={idx}
                    className="example-question"
                    onClick={() => setInputValue(question)}
                    whileHover={{ scale: 1.02, boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}
                    whileTap={{ scale: 0.98 }}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    {question}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          ) : (
            <>
              <AnimatePresence>
                {messages.map((message, index) => (
                  <motion.div
                    key={message.id || index}
                    className={`message-v2 ${message.role}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="message-avatar-v2">
                      {message.role === 'user' ? (
                        <span>{user?.username?.charAt(0).toUpperCase()}</span>
                      ) : (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                          <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
                        </svg>
                      )}
                    </div>
                    <div className="message-content-v2">
                      <div className="message-text-v2">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            code: CodeBlock
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                      {message.documents && message.documents.length > 0 && (
                        <motion.div 
                          className="message-sources-v2"
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          transition={{ delay: 0.2 }}
                        >
                          <details>
                            <summary>
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M14 2V8H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              Nguồn tham khảo ({message.documents.length})
                            </summary>
                            <div className="sources-list-v2">
                              {message.documents.map((doc, idx) => (
                                <motion.div 
                                  key={idx} 
                                  className="source-item-v2"
                                  initial={{ opacity: 0, x: -10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ delay: idx * 0.05 }}
                                >
                                  <div className="source-header-v2">
                                    <strong>{doc.id}</strong>
                                    {doc.score !== undefined && (
                                      <span className="source-score-v2">
                                        {(doc.score * 100).toFixed(1)}%
                                      </span>
                                    )}
                                  </div>
                                  {doc.title && <div className="source-title-v2">{doc.title}</div>}
                                  <div className="source-content-v2">{doc.content}</div>
                                </motion.div>
                              ))}
                            </div>
                          </details>
                        </motion.div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {loading && (
                <motion.div
                  className="message-v2 assistant"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className="message-avatar-v2">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 5C13.66 5 15 6.34 15 8C15 9.66 13.66 11 12 11C10.34 11 9 9.66 9 8C9 6.34 10.34 5 12 5ZM12 19.2C9.5 19.2 7.29 17.92 6 15.98C6.03 13.99 10 12.9 12 12.9C13.99 12.9 17.97 13.99 18 15.98C16.71 17.92 14.5 19.2 12 19.2Z" fill="currentColor"/>
                    </svg>
                  </div>
                  <div className="message-content-v2">
                    <div className="typing-indicator-v2">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </motion.div>
              )}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container-v2">
          <form onSubmit={handleSendMessage} className="input-form-v2">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
              placeholder="Nhập câu hỏi của bạn..."
              disabled={loading}
              rows={1}
            />
            <motion.button
              type="submit"
              disabled={loading || !inputValue.trim()}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </motion.button>
          </form>
          <div className="input-footer-v2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 16V12M12 8H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Chatbot có thể mắc lỗi. Vui lòng kiểm tra thông tin quan trọng.
          </div>
        </div>
      </div>
      
      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {deleteConfirmation && (
          <motion.div 
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setDeleteConfirmation(null)}
          >
            <motion.div 
              className="modal-content"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">SBV Chatbot</div>
              <div className="modal-body">
                Bạn có chắc muốn xóa cuộc trò chuyện này?
              </div>
              <div className="modal-footer">
                <button 
                  className="modal-btn modal-btn-cancel"
                  onClick={() => setDeleteConfirmation(null)}
                >
                  Hủy
                </button>
                <button 
                  className="modal-btn modal-btn-confirm"
                  onClick={confirmDelete}
                >
                  Xóa
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Chat;
