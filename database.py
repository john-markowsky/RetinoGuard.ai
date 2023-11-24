from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.dialects.postgresql import JSONB

# Initialize SQLAlchemy
DATABASE_URL = "sqlite:///./survey_data.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)

# Create a table for user data
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    profession = Column(Integer)  # Assuming profession is stored as an integer
    years_of_experience = Column(String)
    survey_answers = relationship("SurveyAnswers", back_populates="user")

# Create a table for survey responses
class SurveyAnswers(Base):
    __tablename__ = 'survey_answers'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    question_id = Column(String)
    answer = Column(String)
    user = relationship("User", back_populates="survey_answers")

class GradingInterfaceAnswers(Base):
    __tablename__ = 'grading_interface_answers'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    image_id = Column(String)

    # Pre-classified DR rating
    DR_rating = Column(Integer)
    
    # User-provided DR rating
    user_DR_rating = Column(Integer)

    # Accuracy columns
    visual_accuracy = Column(Integer)
    severity_accuracy = Column(Integer)
    
    # Relationship to the User table (if needed)
    user = relationship("User", backref="grading_interface_answers")


# Create all tables
Base.metadata.create_all(bind=engine)

# Initialize session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
