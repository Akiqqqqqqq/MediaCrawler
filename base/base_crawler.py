from abc import ABC, abstractmethod

from base.proxy_account_pool import AccountPool


class AbstractCrawler(ABC):
    @abstractmethod
    def init_config(self, platform: str, login_type: str, account_pool: AccountPool):
        pass

    @abstractmethod
    async def start(self, keywords):
        pass

    @abstractmethod
    async def search(self, keywords) -> list:
        pass


class AbstractLogin(ABC):
    @abstractmethod
    async def begin(self):
        pass

    @abstractmethod
    async def login_by_qrcode(self):
        pass

    @abstractmethod
    async def login_by_mobile(self):
        pass

    @abstractmethod
    async def login_by_cookies(self):
        pass
