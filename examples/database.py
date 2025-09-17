import sys
sys.path.append('..')

from sqlalchemy import Column, Integer, String, text, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
import numpy as np

from db.database import db

# Get the base class from the database singleton
Base = db.get_base()

# Define an Owner model
class Owner(Base):
    __tablename__ = 'owners'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    items = relationship("Item", back_populates="owner")

# Define the Item model with a relationship to Owner
class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    embedding = Column(Vector(3)) # Example vector of 3 dimensions
    owner_id = Column(Integer, ForeignKey('owners.id'))
    owner = relationship("Owner", back_populates="items")

# Create the tables in the database
Base.metadata.create_all(bind=db.engine)

# Get a session
session = db.get_session()

# Create an owner
owner1 = Owner(name='owner1')
session.add(owner1)
session.commit()

# Create a new item and associate it with the owner
item1 = Item(name='item1', embedding=np.array([1, 2, 3]), owner=owner1)
session.add(item1)
session.commit()

# Query the owner and access its items
retrieved_owner = session.query(Owner).filter(Owner.name == 'owner1').first()
if retrieved_owner:
    print(f"Retrieved owner: {retrieved_owner.name}")
    for item in retrieved_owner.items:
        print(f"- Owns item: {item.name}")

# Query the item and access its owner
retrieved_item = session.query(Item).filter(Item.name == 'item1').first()
if retrieved_item:
    print(f"\nRetrieved item: {retrieved_item.name}")
    print(f"Owner: {retrieved_item.owner.name}")
    print(f"Embedding: {retrieved_item.embedding}")

# Example of a vector similarity search
# Find items with embeddings closest to [1, 1, 1]
results = session.query(Item).order_by(Item.embedding.l2_distance([1, 1, 1])).limit(5).all()

print("\nTop 5 closest items to [1, 1, 1]:")
for item in results:
    print(f"- {item.name} (distance: {item.embedding.l2_distance([1, 1, 1])})")


# Close the session
session.close()