"""
PDF文件处理工具

Features:
- PDF完整性校验
- 扫描件检测
- 元数据提取
- 文本内容提取
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import requests
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFValidator:
    """PDF文件验证器"""
    
    @staticmethod
    def validate_download(
        url: str,
        save_path: Path,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> Tuple[bool, str]:
        """
        下载并校验PDF完整性
        
        Args:
            url: PDF下载链接
            save_path: 保存路径
            headers: 请求头
            timeout: 超时时间
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            response = requests.get(
                url,
                headers=headers or {},
                timeout=timeout,
                stream=True
            )
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
                
            # 获取Content-Length
            expected_size = response.headers.get('Content-Length')
            
            # 下载文件
            save_path.parent.mkdir(parents=True, exist_ok=True)
            actual_size = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        actual_size += len(chunk)
                        
            # 大小校验
            if expected_size:
                expected_size = int(expected_size)
                if actual_size != expected_size:
                    save_path.unlink()  # 删除不完整文件
                    return False, f"Size mismatch: {actual_size} != {expected_size}"
                    
            # PDF格式校验
            is_valid, error = PDFValidator.check_pdf_integrity(save_path)
            if not is_valid:
                save_path.unlink()
                return False, f"Invalid PDF: {error}"
                
            return True, ""
            
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            return False, str(e)
    
    @staticmethod
    def check_pdf_integrity(file_path: Path) -> Tuple[bool, str]:
        """
        检查PDF文件完整性
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            # 检查文件头（PDF魔术字节）
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, "Not a PDF file (invalid header)"
                    
                # 检查文件尾
                f.seek(-5, 2)  # 从文件末尾往前5字节
                try:
                    tail = f.read()
                    if b'%%EOF' not in tail:
                        # 有些PDF文件EOF前有空格或换行
                        f.seek(-50, 2)
                        tail = f.read()
                        if b'%%EOF' not in tail:
                            logger.warning(f"PDF may be incomplete (no EOF marker): {file_path}")
                except:
                    pass
                    
            # 尝试用PyPDF2解析
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    _ = len(reader.pages)  # 尝试读取页数
            except ImportError:
                logger.debug("PyPDF2 not installed, skipping deep validation")
            except Exception as e:
                return False, f"PyPDF2 parsing error: {e}"
                
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def is_scanned_pdf(file_path: Path, sample_pages: int = 1) -> bool:
        """
        检测是否为扫描件PDF（图片型）
        
        原理：读取前N页，如果提取不到文本内容，则判定为扫描件
        
        Args:
            file_path: PDF文件路径
            sample_pages: 采样页数
            
        Returns:
            是否为扫描件
        """
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                
                # 检查前N页
                pages_to_check = min(sample_pages, total_pages)
                total_text_length = 0
                
                for i in range(pages_to_check):
                    try:
                        page = reader.pages[i]
                        text = page.extract_text()
                        total_text_length += len(text.strip())
                    except:
                        continue
                        
                # 如果前N页提取的文本少于100字符，判定为扫描件
                if total_text_length < 100:
                    return True
                    
            return False
            
        except ImportError:
            logger.warning("PyPDF2 not installed, cannot detect scanned PDF")
            return False
        except Exception as e:
            logger.error(f"Error detecting scanned PDF: {e}")
            return False
    
    @staticmethod
    def extract_metadata(file_path: Path) -> Dict[str, Any]:
        """
        提取PDF元数据
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            元数据字典
        """
        metadata = {
            'file_size_mb': file_path.stat().st_size / 1024 / 1024,
            'is_valid': False,
            'is_scanned': False,
            'page_count': 0,
            'title': None,
            'author': None,
            'creation_date': None
        }
        
        try:
            # 完整性检查
            is_valid, error = PDFValidator.check_pdf_integrity(file_path)
            metadata['is_valid'] = is_valid
            if not is_valid:
                metadata['error'] = error
                return metadata
                
            # 扫描件检测
            metadata['is_scanned'] = PDFValidator.is_scanned_pdf(file_path)
            
            # 读取PDF信息
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    metadata['page_count'] = len(reader.pages)
                    
                    # 读取文档信息
                    if reader.metadata:
                        metadata['title'] = reader.metadata.get('/Title')
                        metadata['author'] = reader.metadata.get('/Author')
                        metadata['creation_date'] = reader.metadata.get('/CreationDate')
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Error extracting PDF metadata: {e}")
                
        except Exception as e:
            metadata['error'] = str(e)
            
        return metadata


def download_pdf_with_validation(
    url: str,
    save_path: Path,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    check_scanned: bool = True
) -> Dict[str, Any]:
    """
    下载PDF并进行完整性校验
    
    Args:
        url: PDF下载链接
        save_path: 保存路径
        headers: 请求头
        timeout: 超时时间
        check_scanned: 是否检测扫描件
        
    Returns:
        结果字典 {'success': bool, 'metadata': dict, 'error': str}
    """
    result = {
        'success': False,
        'metadata': {},
        'error': None
    }
    
    try:
        # 下载并校验
        success, error = PDFValidator.validate_download(
            url=url,
            save_path=save_path,
            headers=headers,
            timeout=timeout
        )
        
        if not success:
            result['error'] = error
            return result
            
        # 提取元数据
        metadata = PDFValidator.extract_metadata(save_path)
        result['metadata'] = metadata
        
        if check_scanned and metadata.get('is_scanned'):
            logger.warning(f"Downloaded PDF is scanned image: {save_path}")
            metadata['requires_ocr'] = True
            
        result['success'] = True
        logger.info(f"PDF validated successfully: {save_path}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"PDF download/validation failed: {e}")
        
    return result
