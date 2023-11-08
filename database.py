from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Initialize SQLAlchemy
DATABASE_URL = "sqlite:///./survey_data.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)

# Create an inspector object
inspector = inspect(engine)

# Get names of all tables in the database
existing_tables = inspector.get_table_names()

# Create a table for user data
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    profession = Column(Integer)
    years_of_experience = Column(String)
    responses = relationship("Response", back_populates="user")
    annotations = relationship("Annotation", back_populates="user")

# Create a table for survey responses
class Response(Base):
    __tablename__ = 'responses'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ease_of_navigation = Column(Integer)
    user_interface_intuitive = Column(Integer)
    model_accuracy = Column(Integer)
    model_alignment = Column(Integer)
    decision_confidence = Column(Integer)
    grad_cam_usefulness = Column(Integer)
    grad_cam_accuracy = Column(Integer)
    improvements_suggested = Column(String)
    recommend_retinoguard = Column(Integer)
    user = relationship("User", back_populates="responses")

# Create a table for annotations
class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    image_path = Column(String)
    annotated_image_path = Column(String)
    diagnosis = Column(String)
    user = relationship("User", back_populates="annotations")

# Check and create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Initialize session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
