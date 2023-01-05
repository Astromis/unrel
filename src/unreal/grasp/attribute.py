from typing import Callable, Iterable, List, Optional, Set


def _default_attr2text(attr:str, 
                      is_complement:bool = False) -> str:
    if is_complement:
        return f'having an attribute {attr}'
    else:
        return f'a word with an attribute {attr}'


class Attribute():
    
    def __init__(self, 
                 name: str,
                 extraction_function: Callable[[str, List[str]], List[Set[str]]],
                 translation_function: Callable[[str, bool], str] = _default_attr2text,
                 values: Optional[Iterable[str]] = None # Unique binary values of this attribute
                ) -> None:
        self.name = name
        self.extraction_function = extraction_function
        self.translation_function = translation_function
        self.values = values
        
    def __str__(self) -> str:
        if self.values is not None:
            ans = f'{self.name}: {self.values}'
            return ans
        return self.name
    
    def extract(self, text: str, tokens: List[str]) -> List[Set[str]]:
        pre_ans = self.extraction_function(text, tokens)
        ans = [set([f'{self.name}:{item}' for item in t_pre_ans]) for t_pre_ans in pre_ans]
        return ans
        
        
class CustomAttribute(Attribute):
    
    def __init__(self, 
                 name: str,
                 extraction_function: Callable[[str, List[str]], List[Set[str]]],
                 translation_function: Callable[[str, bool], str] = _default_attr2text,
                 values: Optional[Iterable[str]] = None # Unique binary values of this attribute
                ) -> None:
        super().__init__(name, extraction_function, translation_function, values)