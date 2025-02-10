from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

# 数据集
class dataset(Base):
    __tablename__ = 'dataset'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    description = Column(String(200), nullable=False)
    file_path = Column(String(200), nullable=False)
    file_type = Column(String(50), nullable=False)
