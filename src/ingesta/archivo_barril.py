import os
import io
import fitz  # PyMuPDF para extracción de texto
import certifi

from typing import List, Tuple
from datetime import datetime
from pymongo import MongoClient
from utils import segmentar_texto

__all__ = [
    "os", 
    "io", 
    "fitz", 
    "certifi", 
    "List", 
    "Tuple", 
    "datetime", 
    "MongoClient", 
    "segmentar_texto",
]